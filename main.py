import io
import json
import os
import re
import sqlite3
import zipfile
from datetime import datetime
from typing import List

import pandas as pd
from fastapi.concurrency import run_in_threadpool
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    EXPERIENCE_SCORE_WEIGHT,
    EXPORT_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    PROJECTS_SCORE_WEIGHT,
    SKILL_SCORE_WEIGHT,
    UPLOAD_DIR,
    UPLOAD_MAX_SIZE_MB,
    USE_LLM_IN_RANKING,
)
from database import create_tables, get_db
from services.ai_service import (
    build_candidate_embedding_text,
    build_job_embedding_text,
    combine_component_scores,
    extract_candidate_info_async,
    extract_candidate_info_fallback,
    generate_interview_questions_async,
    generate_interview_questions_fallback,
    get_embedding,
    score_candidate_async,
    semantic_score,
)
from services.resume_parser import extract_text

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Recruitment Module", version="1.0.0")


@app.on_event("startup")
def startup():
    create_tables()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)


# ── Schemas ──────────────────────────────────────────────────────────────────


class JobCreate(BaseModel):
    title: str
    description: str
    skills: str
    experience: int = Field(ge=0)


class JobOut(BaseModel):
    id: int
    title: str
    description: str
    skills: str
    experience: int
    created_at: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

SUPPORTED_EXT = (".pdf", ".docx", ".txt")


def _is_supported(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_EXT)


async def _read_with_size_limit(upload: UploadFile) -> bytes:
    """Read upload safely and reject payloads above configured limit."""
    data = await upload.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload.filename or 'unknown'}' exceeds {UPLOAD_MAX_SIZE_MB} MB limit",
        )
    return data


def _embedding_to_db(vector: list[float]) -> str | None:
    if not vector:
        return None
    return json.dumps(vector)


def _embedding_from_db(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return []


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: object, default: float = 0.0) -> float:
    parsed = _to_float(value, default)
    return round(max(0.0, min(100.0, parsed)), 2)


async def _process_single_resume(
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    job_id: int,
    data: bytes,
    filename: str,
) -> dict:
    """Extract text → call AI → persist candidate row. Returns summary dict."""

    raw_text = await run_in_threadpool(extract_text, data, filename)
    if not raw_text or len(raw_text) < 50:
        return {"file": filename, "error": "Could not extract enough text"}

    parser_mode = "ai"
    parser_warning: str | None = None
    try:
        info = await extract_candidate_info_async(raw_text)
    except Exception as exc:
        info = extract_candidate_info_fallback(raw_text)
        parser_mode = "heuristic"
        parser_warning = f"AI extraction failed, heuristic parser used: {exc}"

    if not (info.get("name") or "").strip():
        stem = os.path.splitext(filename)[0]
        fallback_name = re.sub(r"[_\-.]+", " ", stem).strip()
        info["name"] = fallback_name[:80]

    embedding_json = None
    try:
        embedding_text = build_candidate_embedding_text(
            {
                "name": info.get("name", ""),
                "total_experience": info.get("total_experience", ""),
                "skills": info.get("skills", ""),
                "education": info.get("education", ""),
                "certifications": info.get("certifications", ""),
                "summary": info.get("summary", ""),
                "raw_text": raw_text,
            }
        )
        cand_embedding = await run_in_threadpool(
            get_embedding, embedding_text, is_query=False
        )
        embedding_json = _embedding_to_db(cand_embedding)
    except Exception:
        embedding_json = None

    cur.execute(
        """
        INSERT INTO candidates
            (job_id, file_name, raw_text, embedding, name, email, phone,
             total_experience, skills, education, certifications, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            filename,
            raw_text,
            embedding_json,
            info.get("name", ""),
            info.get("email") or "",
            info.get("phone") or "",
            info.get("total_experience", ""),
            info.get("skills", ""),
            info.get("education", ""),
            info.get("certifications", ""),
            info.get("summary", ""),
        ),
    )
    conn.commit()

    return {
        "id": cur.lastrowid,
        "file": filename,
        "name": info.get("name", ""),
        "email": info.get("email", ""),
        "skills": info.get("skills", ""),
        "parser_mode": parser_mode,
        "warning": parser_warning,
    }


# ── Job Endpoints ───────────────────────────────────────────────────────────


@app.post("/jobs/", tags=["Jobs"])
def create_job(body: JobCreate, db: sqlite3.Connection = Depends(get_db)):
    embedding_json = None
    try:
        embedding_text = build_job_embedding_text(
            body.title, body.description, body.skills, body.experience
        )
        embedding_json = _embedding_to_db(get_embedding(embedding_text, is_query=True))
    except Exception:
        embedding_json = None

    cur = db.cursor()
    cur.execute(
        "INSERT INTO jobs (title, description, skills, experience, embedding) VALUES (?, ?, ?, ?, ?)",
        (body.title, body.description, body.skills, body.experience, embedding_json),
    )
    db.commit()
    return {
        "id": cur.lastrowid,
        "title": body.title,
        "skills": body.skills,
        "experience": body.experience,
        "message": "Job created",
    }


@app.get("/jobs/", tags=["Jobs"])
def list_jobs(db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return dict(row)


@app.delete("/jobs/{job_id}", tags=["Jobs"])
def delete_job(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM candidate_questionnaires WHERE job_id = ?", (job_id,))
    cur.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
    cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Job not found")
    return {"message": "Job and its candidates deleted"}


# ── Resume Upload & Extraction ──────────────────────────────────────────────


@app.post("/jobs/{job_id}/upload-resumes/", tags=["Resumes"])
async def upload_resumes(
    job_id: int,
    files: List[UploadFile] = File(...),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Upload one or more resume files (PDF / DOCX / TXT) **or** a single ZIP
    containing resumes.  Each resume is parsed by AI immediately and stored.
    """

    # verify job exists
    if not db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone():
        raise HTTPException(404, "Job not found")

    cur = db.cursor()
    processed: list[dict] = []
    errors: list[dict] = []

    for upload in files:
        fname = upload.filename or "unknown"
        try:
            raw = await _read_with_size_limit(upload)
        except HTTPException as exc:
            errors.append({"file": fname, "error": str(exc.detail)})
            continue

        # ── ZIP handling ────────────────────────────────────────────────
        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        entry = info.filename
                        if info.is_dir() or entry.startswith(("__", ".")):
                            continue
                        if not _is_supported(entry):
                            continue

                        if info.file_size > MAX_UPLOAD_SIZE_BYTES:
                            errors.append(
                                {
                                    "file": os.path.basename(entry),
                                    "error": f"File inside ZIP exceeds {UPLOAD_MAX_SIZE_MB} MB limit",
                                }
                            )
                            continue

                        entry_data = zf.read(info)
                        result = await _process_single_resume(
                            cur, db, job_id, entry_data, os.path.basename(entry)
                        )
                        (errors if "error" in result else processed).append(result)
            except zipfile.BadZipFile:
                errors.append({"file": fname, "error": "Invalid ZIP file"})
            continue

        # ── Single file handling ────────────────────────────────────────
        if not _is_supported(fname):
            errors.append({"file": fname, "error": "Unsupported file type"})
            continue

        result = await _process_single_resume(cur, db, job_id, raw, fname)
        (errors if "error" in result else processed).append(result)

    return {
        "message": f"Processed {len(processed)} resume(s)",
        "processed": processed,
        "errors": errors,
    }


# ── Ranking ─────────────────────────────────────────────────────────────────


@app.post("/jobs/{job_id}/rank-candidates/", tags=["Ranking"])
async def rank_candidates(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Rank candidates using component LLM scores (skills/projects/experience)."""

    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    candidates = db.execute(
        "SELECT * FROM candidates WHERE job_id = ?", (job_id,)
    ).fetchall()
    if not candidates:
        raise HTTPException(404, "No candidates found for this job")

    cur = db.cursor()
    results: list[dict] = []

    job_embedding = _embedding_from_db(job["embedding"])
    if not job_embedding:
        try:
            job_text = build_job_embedding_text(
                job["title"],
                job["description"],
                job["skills"] if "skills" in job.keys() else "",
                job["experience"] if "experience" in job.keys() else 0,
            )
            job_embedding = await run_in_threadpool(get_embedding, job_text, is_query=True)
            cur.execute(
                "UPDATE jobs SET embedding = ? WHERE id = ?",
                (_embedding_to_db(job_embedding), job_id),
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to generate job embedding: {exc}")

    for cand in candidates:
        c = dict(cand)

        try:
            cand_embedding = _embedding_from_db(c.get("embedding"))
            if not cand_embedding:
                cand_text = build_candidate_embedding_text(c)
                cand_embedding = await run_in_threadpool(
                    get_embedding, cand_text, is_query=False
                )
                cur.execute(
                    "UPDATE candidates SET embedding = ? WHERE id = ?",
                    (_embedding_to_db(cand_embedding), c["id"]),
                )

            semantic = round(semantic_score(job_embedding, cand_embedding), 2)
            skill_score = _clamp_score(c.get("skill_score"))
            projects_score = _clamp_score(c.get("projects_score"))
            experience_score = _clamp_score(c.get("experience_score"))
            total_score = _clamp_score(c.get("total_score"), _to_float(c.get("score")))
            if total_score <= 0 and any(
                score > 0 for score in (skill_score, projects_score, experience_score)
            ):
                total_score = combine_component_scores(
                    skill_score,
                    projects_score,
                    experience_score,
                    skill_weight=SKILL_SCORE_WEIGHT,
                    projects_weight=PROJECTS_SCORE_WEIGHT,
                    experience_weight=EXPERIENCE_SCORE_WEIGHT,
                )
            had_prior_scores = any(
                score > 0
                for score in (skill_score, projects_score, experience_score, total_score)
            )
            needs_review = 0
            review_reason: str | None = None
            reasoning = "Component-based ranking generated from LLM evaluation."

            if USE_LLM_IN_RANKING:
                try:
                    ai = await score_candidate_async(
                        job["description"],
                        job["skills"] if "skills" in job.keys() else "",
                        job["experience"] if "experience" in job.keys() else 0,
                        c,
                    )
                    skill_score = _clamp_score(ai.get("skill_score"))
                    projects_score = _clamp_score(ai.get("projects_score"))
                    experience_score = _clamp_score(ai.get("experience_score"))
                    total_score = combine_component_scores(
                        skill_score,
                        projects_score,
                        experience_score,
                        skill_weight=SKILL_SCORE_WEIGHT,
                        projects_weight=PROJECTS_SCORE_WEIGHT,
                        experience_weight=EXPERIENCE_SCORE_WEIGHT,
                    )
                    llm_reasoning = ai.get("reasoning", "")
                    if llm_reasoning:
                        reasoning = llm_reasoning
                    needs_review = 0
                    review_reason = None
                except Exception as llm_exc:
                    if not had_prior_scores:
                        skill_score = 0.0
                        projects_score = 0.0
                        experience_score = 0.0
                        total_score = 0.0
                    needs_review = 1
                    review_reason = f"LLM component scoring failed: {llm_exc}"
                    if had_prior_scores:
                        reasoning = (
                            "Candidate flagged for manual review because LLM component "
                            "scoring failed. Existing scores were preserved."
                        )
                    else:
                        reasoning = (
                            "Candidate flagged for manual review because LLM component "
                            "scoring failed before component scores were available."
                        )
            else:
                needs_review = 1
                review_reason = "LLM component scoring is disabled by configuration."
                reasoning = (
                    "Candidate flagged for manual review because LLM component scoring "
                    "is disabled."
                )

            cur.execute(
                """UPDATE candidates
                   SET semantic_score = ?, skill_score = ?, projects_score = ?,
                       experience_score = ?, total_score = ?, score = ?,
                       score_reasoning = ?, needs_review = ?, review_reason = ?
                   WHERE id = ?""",
                (
                    semantic,
                    skill_score,
                    projects_score,
                    experience_score,
                    total_score,
                    total_score,  # Backward-compat alias for existing clients.
                    reasoning,
                    needs_review,
                    review_reason,
                    c["id"],
                ),
            )
            results.append(
                {
                    "id": c["id"],
                    "name": c["name"],
                    "semantic_score": semantic,
                    "skill_score": skill_score,
                    "projects_score": projects_score,
                    "experience_score": experience_score,
                    "total_score": total_score,
                    "score": total_score,
                    "needs_review": bool(needs_review),
                    "review_reason": review_reason,
                    "reasoning": reasoning,
                }
            )
        except Exception as exc:
            err = str(exc)
            cur.execute(
                """UPDATE candidates
                   SET semantic_score = 0, skill_score = 0, projects_score = 0,
                       experience_score = 0, total_score = 0, score = 0,
                       score_reasoning = ?, needs_review = 1, review_reason = ?
                   WHERE id = ?""",
                (
                    f"Ranking failed: {err}",
                    f"Ranking pipeline failed: {err}",
                    c["id"],
                ),
            )
            results.append({"id": c["id"], "name": c["name"], "error": err})

    # assign ranks (1 = best)
    ranked_ids = db.execute(
        "SELECT id FROM candidates WHERE job_id = ? ORDER BY total_score DESC, semantic_score DESC",
        (job_id,),
    ).fetchall()
    for idx, row in enumerate(ranked_ids, 1):
        cur.execute("UPDATE candidates SET rank = ? WHERE id = ?", (idx, row["id"]))

    db.commit()

    # return final ranked list
    final = db.execute(
        """SELECT id, name, email, phone, total_experience, skills, education,
                  certifications, semantic_score, skill_score, projects_score,
                  experience_score, total_score, score, needs_review, review_reason,
                  rank, score_reasoning
           FROM candidates WHERE job_id = ? ORDER BY rank""",
        (job_id,),
    ).fetchall()

    return {"message": f"Ranked {len(final)} candidate(s)", "candidates": [dict(r) for r in final]}


# ── Interview Questionnaires ────────────────────────────────────────────────


@app.post("/jobs/{job_id}/generate-questionnaires/", tags=["Interview"])
async def generate_questionnaires(
    job_id: int,
    top_n: int = Query(5, ge=1, le=50),
    questions_per_candidate: int = Query(8, ge=3, le=20),
    db: sqlite3.Connection = Depends(get_db),
):
    """Generate personalized interview questions for top-ranked candidates."""

    job_row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job_row:
        raise HTTPException(404, "Job not found")

    candidate_count = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE job_id = ?", (job_id,)
    ).fetchone()[0]
    if candidate_count == 0:
        raise HTTPException(404, "No candidates found for this job")

    ranked_count = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE job_id = ? AND rank > 0", (job_id,)
    ).fetchone()[0]
    if ranked_count == 0:
        await rank_candidates(job_id, db)

    top_rows = db.execute(
        """SELECT * FROM candidates
           WHERE job_id = ?
           ORDER BY rank ASC, total_score DESC, semantic_score DESC
           LIMIT ?""",
        (job_id, top_n),
    ).fetchall()
    if not top_rows:
        raise HTTPException(404, "No ranked candidates available")

    cur = db.cursor()
    generated: list[dict] = []
    job = dict(job_row)

    for candidate_row in top_rows:
        candidate = dict(candidate_row)
        generation_mode = "ai"
        warning: str | None = None

        try:
            questions = await generate_interview_questions_async(
                job,
                candidate,
                questions_per_candidate,
            )
        except Exception as exc:
            questions = generate_interview_questions_fallback(
                job,
                candidate,
                questions_per_candidate,
            )
            generation_mode = "fallback"
            warning = f"AI questionnaire generation failed, fallback used: {exc}"

        cur.execute(
            "DELETE FROM candidate_questionnaires WHERE candidate_id = ?",
            (candidate["id"],),
        )
        for idx, item in enumerate(questions, 1):
            cur.execute(
                """INSERT INTO candidate_questionnaires
                       (job_id, candidate_id, question_order, question_text, focus_area,
                        difficulty, reasoning, generation_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    candidate["id"],
                    idx,
                    item.get("question", "").strip(),
                    item.get("focus_area", "general"),
                    item.get("difficulty", "medium"),
                    item.get("reason", ""),
                    generation_mode,
                ),
            )

        generated.append(
            {
                "candidate_id": candidate["id"],
                "name": candidate.get("name", ""),
                "rank": candidate.get("rank", 0),
                "total_score": candidate.get("total_score", 0),
                "needs_review": bool(candidate.get("needs_review", 0)),
                "generation_mode": generation_mode,
                "warning": warning,
                "questions_count": len(questions),
                "questions": questions,
            }
        )

    db.commit()
    return {
        "message": f"Generated questionnaires for {len(generated)} candidate(s)",
        "job_id": job_id,
        "top_n": top_n,
        "questions_per_candidate": questions_per_candidate,
        "candidates": generated,
    }


@app.get("/jobs/{job_id}/questionnaires/", tags=["Interview"])
def list_job_questionnaires(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """SELECT q.candidate_id, c.name, c.email, c.rank, c.total_score, c.needs_review,
                  q.question_order, q.question_text, q.focus_area, q.difficulty,
                  q.reasoning, q.generation_mode, q.created_at
           FROM candidate_questionnaires q
           JOIN candidates c ON c.id = q.candidate_id
           WHERE q.job_id = ?
           ORDER BY c.rank ASC, q.candidate_id ASC, q.question_order ASC""",
        (job_id,),
    ).fetchall()
    if not rows:
        raise HTTPException(404, "No questionnaires found for this job")

    grouped: dict[int, dict] = {}
    for row in rows:
        item = dict(row)
        candidate_id = item["candidate_id"]
        if candidate_id not in grouped:
            grouped[candidate_id] = {
                "candidate_id": candidate_id,
                "name": item.get("name", ""),
                "email": item.get("email", ""),
                "rank": item.get("rank", 0),
                "total_score": item.get("total_score", 0),
                "needs_review": bool(item.get("needs_review", 0)),
                "questions": [],
            }
        grouped[candidate_id]["questions"].append(
            {
                "order": item.get("question_order", 0),
                "question": item.get("question_text", ""),
                "focus_area": item.get("focus_area", "general"),
                "difficulty": item.get("difficulty", "medium"),
                "reason": item.get("reasoning", ""),
                "generation_mode": item.get("generation_mode", "ai"),
                "created_at": item.get("created_at"),
            }
        )

    return list(grouped.values())


@app.get("/candidates/{candidate_id}/questionnaire/", tags=["Interview"])
def get_candidate_questionnaire(
    candidate_id: int,
    db: sqlite3.Connection = Depends(get_db),
):
    candidate = db.execute(
        """SELECT id, job_id, name, email, rank, total_score, needs_review
           FROM candidates WHERE id = ?""",
        (candidate_id,),
    ).fetchone()
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    rows = db.execute(
        """SELECT question_order, question_text, focus_area, difficulty,
                  reasoning, generation_mode, created_at
           FROM candidate_questionnaires
           WHERE candidate_id = ?
           ORDER BY question_order ASC""",
        (candidate_id,),
    ).fetchall()
    if not rows:
        raise HTTPException(404, "No questionnaire found for this candidate")

    candidate_data = dict(candidate)
    return {
        "candidate_id": candidate_data["id"],
        "job_id": candidate_data["job_id"],
        "name": candidate_data.get("name", ""),
        "email": candidate_data.get("email", ""),
        "rank": candidate_data.get("rank", 0),
        "total_score": candidate_data.get("total_score", 0),
        "needs_review": bool(candidate_data.get("needs_review", 0)),
        "questions": [
            {
                "order": item["question_order"],
                "question": item["question_text"],
                "focus_area": item["focus_area"] or "general",
                "difficulty": item["difficulty"] or "medium",
                "reason": item["reasoning"] or "",
                "generation_mode": item["generation_mode"] or "ai",
                "created_at": item["created_at"],
            }
            for item in rows
        ],
    }


# ── Master Pipeline ──────────────────────────────────────────────────────────


@app.post(
    "/jobs/{job_id}/run-pipeline/",
    tags=["Pipeline"],
    description=(
        "Master endpoint: optional resume upload -> ranking -> questionnaire generation. "
        "Job creation is intentionally outside this pipeline."
    ),
)
async def run_master_pipeline(
    job_id: int,
    top_n: int = Query(5, ge=1, le=50),
    questions_per_candidate: int = Query(8, ge=3, le=20),
    files: List[UploadFile] | None = File(default=None),
    db: sqlite3.Connection = Depends(get_db),
):
    """
    Run end-to-end recruitment flow for an existing job:
      1) Optional resume upload/extraction
      2) Candidate ranking
      3) Personalized questionnaire generation for top candidates
    """

    job_exists = db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job_exists:
        raise HTTPException(404, "Job not found")

    upload_result: dict | None = None
    uploaded_files = [f for f in (files or []) if f is not None]
    if uploaded_files:
        upload_result = await upload_resumes(job_id, uploaded_files, db)

    candidate_count = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]
    if candidate_count == 0:
        raise HTTPException(
            404,
            "No candidates found for this job. Upload resumes first or pass files to this endpoint.",
        )

    ranking_result = await rank_candidates(job_id, db)
    questionnaire_result = await generate_questionnaires(
        job_id=job_id,
        top_n=top_n,
        questions_per_candidate=questions_per_candidate,
        db=db,
    )

    return {
        "message": "Master pipeline completed",
        "job_id": job_id,
        "uploaded": upload_result,
        "ranking": {
            "message": ranking_result.get("message"),
            "candidates_count": len(ranking_result.get("candidates", [])),
        },
        "questionnaires": {
            "message": questionnaire_result.get("message"),
            "top_n": questionnaire_result.get("top_n"),
            "questions_per_candidate": questionnaire_result.get("questions_per_candidate"),
            "candidates_count": len(questionnaire_result.get("candidates", [])),
            "candidates": questionnaire_result.get("candidates", []),
        },
    }


# ── Candidate Endpoints ─────────────────────────────────────────────────────


@app.get("/jobs/{job_id}/candidates/", tags=["Candidates"])
def list_candidates(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """SELECT id, file_name, name, email, phone, total_experience,
                  skills, education, certifications, summary,
                  semantic_score, skill_score, projects_score, experience_score,
                  total_score, score, needs_review, review_reason,
                  rank, score_reasoning, created_at
           FROM candidates WHERE job_id = ? ORDER BY rank ASC, created_at DESC""",
        (job_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/candidates/{candidate_id}", tags=["Candidates"])
def get_candidate(candidate_id: int, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Candidate not found")
    return dict(row)


@app.delete("/candidates/{candidate_id}", tags=["Candidates"])
def delete_candidate(candidate_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM candidate_questionnaires WHERE candidate_id = ?", (candidate_id,))
    cur.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Candidate not found")
    return {"message": "Candidate deleted"}

@app.delete("/candidates/", tags=["Candidates"], description="**Warning:** This will permanently delete all candidate records from the database.")
def delete_all_candidates(db: sqlite3.Connection = Depends(get_db)):
    
    cur = db.cursor()

    cur.execute('SELECT COUNT(*) FROM candidates')
    
    count = cur.fetchone()[0]

    if count == 0:
        raise HTTPException(404, "Candidate not found")

    cur.execute("DELETE FROM candidate_questionnaires")
    cur.execute("DELETE FROM candidates")
    db.commit()

    return {"message": f"{count} Candidates deleted"}



# ── Excel Export ─────────────────────────────────────────────────────────────


@app.get("/jobs/{job_id}/export-excel/", tags=["Export"])
def export_excel(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Download an Excel file with ranked candidate data for a job."""

    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    rows = db.execute(
        """SELECT name, email, phone, total_experience, skills, education,
                  certifications, semantic_score, skill_score, projects_score,
                  experience_score, total_score, needs_review, review_reason,
                  rank, score_reasoning
           FROM candidates WHERE job_id = ? ORDER BY rank ASC""",
        (job_id,),
    ).fetchall()

    if not rows:
        raise HTTPException(404, "No candidates to export")

    df = pd.DataFrame([dict(r) for r in rows])
    df.columns = [
        "Name", "Email", "Phone", "Total Experience", "Key Skills",
        "Education", "Certifications", "Semantic Score (out of 100)",
        "Skill Score (out of 100)", "Projects Score (out of 100)",
        "Experience Score (out of 100)", "Total Score (out of 100)", "Needs Review",
        "Review Reason", "Rank", "Score Reasoning",
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"candidates_job_{job_id}_{ts}.xlsx"
    fpath = os.path.join(EXPORT_DIR, fname)

    with pd.ExcelWriter(fpath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")

        # auto-fit column widths
        ws = writer.sheets["Candidates"]
        for col_cells in ws.columns:
            length = max(len(str(c.value or "")) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(length + 3, 55)

    return FileResponse(
        fpath,
        filename=fname,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000)
