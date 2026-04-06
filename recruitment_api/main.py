import io
import json
import os
import sqlite3
import zipfile
from datetime import datetime
from typing import List

import pandas as pd
from fastapi.concurrency import run_in_threadpool
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    EXPORT_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOAD_DIR,
    UPLOAD_MAX_SIZE_MB,
    USE_LLM_IN_RANKING,
)
from database import create_tables, get_db
from services.ai_service import (
    build_candidate_embedding_text,
    build_job_embedding_text,
    combine_scores,
    extract_candidate_info_async,
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

    try:
        info = await extract_candidate_info_async(raw_text)
    except Exception as exc:
        return {"file": filename, "error": f"AI extraction failed: {exc}"}

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
    """Rank candidates using semantic similarity, optionally blended with LLM score."""

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
            llm_score_val: float | None = None
            reasoning = "Semantic ranking generated from embedding similarity."

            if USE_LLM_IN_RANKING:
                ai = await score_candidate_async(
                    job["description"],
                    job["skills"] if "skills" in job.keys() else "",
                    job["experience"] if "experience" in job.keys() else 0,
                    c,
                )
                llm_score_val = float(ai.get("score", 0))
                llm_reasoning = ai.get("reasoning", "")
                if llm_reasoning:
                    reasoning = llm_reasoning

            final_score = combine_scores(semantic, llm_score_val)
            cur.execute(
                """UPDATE candidates
                   SET semantic_score = ?, score = ?, score_reasoning = ?
                   WHERE id = ?""",
                (semantic, final_score, reasoning, c["id"]),
            )
            results.append(
                {
                    "id": c["id"],
                    "name": c["name"],
                    "semantic_score": semantic,
                    "score": final_score,
                    "reasoning": reasoning,
                }
            )
        except Exception as exc:
            err = str(exc)
            cur.execute(
                """UPDATE candidates
                   SET semantic_score = 0, score = 0, score_reasoning = ?
                   WHERE id = ?""",
                (f"Ranking failed: {err}", c["id"]),
            )
            results.append({"id": c["id"], "name": c["name"], "error": err})

    # assign ranks (1 = best)
    ranked_ids = db.execute(
        "SELECT id FROM candidates WHERE job_id = ? ORDER BY score DESC", (job_id,)
    ).fetchall()
    for idx, row in enumerate(ranked_ids, 1):
        cur.execute("UPDATE candidates SET rank = ? WHERE id = ?", (idx, row["id"]))

    db.commit()

    # return final ranked list
    final = db.execute(
        """SELECT id, name, email, phone, total_experience, skills, education,
                  certifications, semantic_score, score, rank, score_reasoning
           FROM candidates WHERE job_id = ? ORDER BY rank""",
        (job_id,),
    ).fetchall()

    return {"message": f"Ranked {len(final)} candidate(s)", "candidates": [dict(r) for r in final]}


# ── Candidate Endpoints ─────────────────────────────────────────────────────


@app.get("/jobs/{job_id}/candidates/", tags=["Candidates"])
def list_candidates(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """SELECT id, file_name, name, email, phone, total_experience,
                  skills, education, certifications, summary,
                  semantic_score, score, rank, score_reasoning, created_at
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
                  certifications, semantic_score, score, rank, score_reasoning
           FROM candidates WHERE job_id = ? ORDER BY rank ASC""",
        (job_id,),
    ).fetchall()

    if not rows:
        raise HTTPException(404, "No candidates to export")

    df = pd.DataFrame([dict(r) for r in rows])
    df.columns = [
        "Name", "Email", "Phone", "Total Experience", "Key Skills",
        "Education", "Certifications", "Semantic Score", "Final Score", "Rank", "Score Reasoning",
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
