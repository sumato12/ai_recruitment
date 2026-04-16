import asyncio
import io
import json
import logging
import os
import re
import sqlite3
import zipfile

from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from config import (
    EXPERIENCE_SCORE_WEIGHT,
    MAX_UPLOAD_SIZE_BYTES,
    PROJECTS_SCORE_WEIGHT,
    QUESTIONNAIRE_CONCURRENCY,
    RANKING_CONCURRENCY,
    RESUME_PROCESSING_CONCURRENCY,
    SKILL_SCORE_WEIGHT,
    UPLOAD_MAX_SIZE_MB,
    USE_LLM_IN_RANKING,
)
from services.ai_service import (
    build_candidate_embedding_text,
    build_job_embedding_text,
    combine_component_scores,
    extract_candidate_info_async,
    extract_candidate_info_fallback,
    generate_interview_questions_async,
    generate_interview_questions_by_topic_async,
    generate_interview_questions_by_topic_fallback,
    generate_interview_questions_fallback,
    get_embedding,
    score_candidate_async,
    semantic_score,
)
from services.resume_parser import extract_text

logger = logging.getLogger(__name__)

SUPPORTED_EXT = (".pdf", ".docx", ".txt")


def is_supported(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_EXT)


async def read_with_size_limit(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload.filename or 'unknown'}' exceeds {UPLOAD_MAX_SIZE_MB} MB limit",
        )
    return data


def embedding_to_db(vector: list[float]) -> str | None:
    if not vector:
        return None
    return json.dumps(vector)


def embedding_from_db(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return []


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_score(value: object, default: float = 0.0) -> float:
    parsed = to_float(value, default)
    return round(max(0.0, min(100.0, parsed)), 2)


async def extract_resume_details(data: bytes, filename: str) -> dict:
    try:
        raw_text = await run_in_threadpool(extract_text, data, filename)
    except Exception as exc:
        logger.warning("Text extraction failed for '%s': %s", filename, exc)
        return {"file": filename, "error": f"Could not extract text: {exc}"}

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
        logger.warning("AI extraction failed for '%s'; using heuristic parser: %s", filename, exc)

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
        embedding_json = embedding_to_db(cand_embedding)
    except Exception as exc:
        logger.warning("Embedding generation failed for '%s': %s", filename, exc)

    return {
        "file": filename,
        "raw_text": raw_text,
        "embedding_json": embedding_json,
        "name": info.get("name", ""),
        "email": info.get("email") or "",
        "phone": info.get("phone") or "",
        "total_experience": info.get("total_experience", ""),
        "skills": info.get("skills", ""),
        "education": info.get("education", ""),
        "certifications": info.get("certifications", ""),
        "summary": info.get("summary", ""),
        "parser_mode": parser_mode,
        "warning": parser_warning,
    }


def insert_candidate_record(
    cur: sqlite3.Cursor,
    job_id: int,
    parsed_resume: dict,
) -> int:
    cur.execute(
        """
        INSERT INTO candidates
            (job_id, file_name, raw_text, embedding, name, email, phone,
             total_experience, skills, education, certifications, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            parsed_resume["file"],
            parsed_resume["raw_text"],
            parsed_resume["embedding_json"],
            parsed_resume["name"],
            parsed_resume["email"],
            parsed_resume["phone"],
            parsed_resume["total_experience"],
            parsed_resume["skills"],
            parsed_resume["education"],
            parsed_resume["certifications"],
            parsed_resume["summary"],
        ),
    )
    return cur.lastrowid


async def upload_resumes_for_job(
    job_id: int,
    files: list[UploadFile],
    db: sqlite3.Connection,
) -> dict:
    if not db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone():
        raise HTTPException(404, "Job not found")

    cur = db.cursor()
    processed: list[dict] = []
    errors: list[dict] = []
    pending_resumes: list[tuple[str, bytes]] = []

    for upload in files:
        fname = upload.filename or "unknown"
        try:
            raw = await read_with_size_limit(upload)
        except HTTPException as exc:
            errors.append({"file": fname, "error": str(exc.detail)})
            continue

        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        entry = info.filename
                        if info.is_dir() or entry.startswith(("__", ".")):
                            continue
                        if not is_supported(entry):
                            continue
                        if info.file_size > MAX_UPLOAD_SIZE_BYTES:
                            errors.append(
                                {
                                    "file": os.path.basename(entry),
                                    "error": f"File inside ZIP exceeds {UPLOAD_MAX_SIZE_MB} MB limit",
                                }
                            )
                            continue
                        try:
                            pending_resumes.append((os.path.basename(entry), zf.read(info)))
                        except Exception as exc:
                            errors.append(
                                {
                                    "file": os.path.basename(entry),
                                    "error": f"Failed reading ZIP entry: {exc}",
                                }
                            )
            except zipfile.BadZipFile:
                errors.append({"file": fname, "error": "Invalid ZIP file"})
            continue

        if not is_supported(fname):
            errors.append({"file": fname, "error": "Unsupported file type"})
            continue

        pending_resumes.append((fname, raw))

    async def _extract_with_limit(
        filename: str,
        data: bytes,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        async with semaphore:
            try:
                return await extract_resume_details(data, filename)
            except Exception as exc:
                logger.exception("Unexpected extraction failure for '%s': %s", filename, exc)
                return {"file": filename, "error": f"Unexpected extraction failure: {exc}"}

    if pending_resumes:
        extract_semaphore = asyncio.Semaphore(RESUME_PROCESSING_CONCURRENCY)
        extracted_resumes = await asyncio.gather(
            *[
                _extract_with_limit(filename, data, extract_semaphore)
                for filename, data in pending_resumes
            ],
            return_exceptions=True,
        )

        for parsed in extracted_resumes:
            if isinstance(parsed, Exception):
                errors.append(
                    {
                        "file": "unknown",
                        "error": f"Unhandled extraction failure: {parsed}",
                    }
                )
                continue
            if "error" in parsed:
                errors.append({"file": parsed.get("file", "unknown"), "error": parsed["error"]})
                continue

            try:
                candidate_id = insert_candidate_record(cur, job_id, parsed)
            except Exception as exc:
                logger.exception("Failed to save parsed resume '%s': %s", parsed.get("file"), exc)
                errors.append(
                    {
                        "file": parsed.get("file", "unknown"),
                        "error": f"Failed to store candidate record: {exc}",
                    }
                )
                continue

            processed.append(
                {
                    "id": candidate_id,
                    "file": parsed["file"],
                    "name": parsed["name"],
                    "email": parsed["email"],
                    "skills": parsed["skills"],
                    "parser_mode": parsed["parser_mode"],
                    "warning": parsed["warning"],
                }
            )

        if processed:
            db.commit()

    return {
        "message": f"Processed {len(processed)} resume(s)",
        "processed": processed,
        "errors": errors,
    }


async def _score_candidate_for_job(
    candidate: dict,
    job: dict,
    job_embedding: list[float],
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        try:
            cand_embedding = embedding_from_db(candidate.get("embedding"))
            embedding_json = None
            if not cand_embedding:
                cand_text = build_candidate_embedding_text(candidate)
                cand_embedding = await run_in_threadpool(
                    get_embedding, cand_text, is_query=False
                )
                embedding_json = embedding_to_db(cand_embedding)

            semantic = round(semantic_score(job_embedding, cand_embedding), 2)
            skill_score = clamp_score(candidate.get("skill_score"))
            projects_score = clamp_score(candidate.get("projects_score"))
            experience_score = clamp_score(candidate.get("experience_score"))
            total_score = clamp_score(
                candidate.get("total_score"),
                to_float(candidate.get("score")),
            )
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
                        job.get("description", ""),
                        job.get("skills", ""),
                        job.get("experience", 0),
                        candidate,
                    )
                    skill_score = clamp_score(ai.get("skill_score"))
                    projects_score = clamp_score(ai.get("projects_score"))
                    experience_score = clamp_score(ai.get("experience_score"))
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
                    logger.warning(
                        "LLM component scoring failed for candidate '%s': %s",
                        candidate.get("id"),
                        llm_exc,
                    )
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

            return {
                "id": candidate["id"],
                "name": candidate.get("name", ""),
                "embedding_json": embedding_json,
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
        except Exception as exc:
            return {"id": candidate["id"], "name": candidate.get("name", ""), "error": str(exc)}


async def rank_candidates_for_job(job_id: int, db: sqlite3.Connection) -> dict:
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    candidates = db.execute(
        "SELECT * FROM candidates WHERE job_id = ?", (job_id,)
    ).fetchall()
    if not candidates:
        raise HTTPException(404, "No candidates found for this job")

    cur = db.cursor()
    job_embedding = embedding_from_db(job["embedding"])
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
                (embedding_to_db(job_embedding), job_id),
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to generate job embedding: {exc}")

    job_data = dict(job)
    candidate_rows = [dict(row) for row in candidates]
    ranking_semaphore = asyncio.Semaphore(RANKING_CONCURRENCY)
    scored_candidates = await asyncio.gather(
        *[
            _score_candidate_for_job(
                candidate,
                job_data,
                job_embedding,
                ranking_semaphore,
            )
            for candidate in candidate_rows
        ]
    )

    for ranked in scored_candidates:
        candidate_id = ranked["id"]
        if "error" in ranked:
            err = ranked["error"]
            cur.execute(
                """UPDATE candidates
                   SET semantic_score = 0, skill_score = 0, projects_score = 0,
                       experience_score = 0, total_score = 0, score = 0,
                       score_reasoning = ?, needs_review = 1, review_reason = ?
                   WHERE id = ?""",
                (
                    f"Ranking failed: {err}",
                    f"Ranking pipeline failed: {err}",
                    candidate_id,
                ),
            )
            continue

        if ranked.get("embedding_json"):
            cur.execute(
                "UPDATE candidates SET embedding = ? WHERE id = ?",
                (ranked["embedding_json"], candidate_id),
            )

        cur.execute(
            """UPDATE candidates
               SET semantic_score = ?, skill_score = ?, projects_score = ?,
                   experience_score = ?, total_score = ?, score = ?,
                   score_reasoning = ?, needs_review = ?, review_reason = ?
               WHERE id = ?""",
            (
                ranked["semantic_score"],
                ranked["skill_score"],
                ranked["projects_score"],
                ranked["experience_score"],
                ranked["total_score"],
                ranked["score"],
                ranked["reasoning"],
                int(ranked["needs_review"]),
                ranked["review_reason"],
                candidate_id,
            ),
        )

    ranked_ids = db.execute(
        "SELECT id FROM candidates WHERE job_id = ? ORDER BY total_score DESC, semantic_score DESC",
        (job_id,),
    ).fetchall()
    for idx, row in enumerate(ranked_ids, 1):
        cur.execute("UPDATE candidates SET rank = ? WHERE id = ?", (idx, row["id"]))

    db.commit()
    final = db.execute(
        """SELECT id, name, email, phone, total_experience, skills, education,
                  certifications, semantic_score, skill_score, projects_score,
                  experience_score, total_score, score, needs_review, review_reason,
                  rank, score_reasoning
           FROM candidates WHERE job_id = ? ORDER BY rank""",
        (job_id,),
    ).fetchall()

    return {"message": f"Ranked {len(final)} candidate(s)", "candidates": [dict(r) for r in final]}


async def ensure_all_candidates_ranked(job_id: int, db: sqlite3.Connection) -> None:
    candidate_count = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]
    if candidate_count == 0:
        raise HTTPException(404, "No candidates found for this job")

    ranked_count = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE job_id = ? AND rank > 0",
        (job_id,),
    ).fetchone()[0]
    if ranked_count < candidate_count:
        await rank_candidates_for_job(job_id, db)


async def _build_questionnaire_for_candidate(
    job: dict,
    candidate: dict,
    questions_per_candidate: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
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
            logger.warning(
                "AI questionnaire generation failed for candidate '%s': %s",
                candidate.get("id"),
                exc,
            )

        return {
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


async def _build_topic_questionnaire_for_candidate(
    job: dict,
    candidate: dict,
    topic_counts: dict[str, int],
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        generation_mode = "ai"
        warning: str | None = None

        try:
            questions = await generate_interview_questions_by_topic_async(
                job,
                candidate,
                topic_counts,
            )
        except Exception as exc:
            questions = generate_interview_questions_by_topic_fallback(
                job,
                candidate,
                topic_counts,
            )
            generation_mode = "fallback"
            warning = f"AI topic questionnaire generation failed, fallback used: {exc}"
            logger.warning(
                "AI topic questionnaire generation failed for candidate '%s': %s",
                candidate.get("id"),
                exc,
            )

        return {
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


async def generate_questionnaires_for_job(
    job_id: int,
    top_n: int,
    questions_per_candidate: int,
    db: sqlite3.Connection,
) -> dict:
    job_row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job_row:
        raise HTTPException(404, "Job not found")

    await ensure_all_candidates_ranked(job_id, db)

    top_rows = db.execute(
        """SELECT * FROM candidates
           WHERE job_id = ? AND rank > 0
           ORDER BY rank ASC, total_score DESC, semantic_score DESC
           LIMIT ?""",
        (job_id, top_n),
    ).fetchall()
    if not top_rows:
        raise HTTPException(404, "No ranked candidates available")

    cur = db.cursor()
    job = dict(job_row)
    questionnaire_semaphore = asyncio.Semaphore(QUESTIONNAIRE_CONCURRENCY)
    generated = await asyncio.gather(
        *[
            _build_questionnaire_for_candidate(
                job,
                dict(candidate_row),
                questions_per_candidate,
                questionnaire_semaphore,
            )
            for candidate_row in top_rows
        ]
    )

    for item in generated:
        cur.execute(
            "DELETE FROM candidate_questionnaires WHERE candidate_id = ?",
            (item["candidate_id"],),
        )
        for idx, question in enumerate(item["questions"], 1):
            cur.execute(
                """INSERT INTO candidate_questionnaires
                       (job_id, candidate_id, question_order, question_text, focus_area,
                        difficulty, reasoning, generation_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    item["candidate_id"],
                    idx,
                    question.get("question", "").strip(),
                    question.get("focus_area", "general"),
                    question.get("difficulty", "medium"),
                    question.get("reason", ""),
                    item["generation_mode"],
                ),
            )

    db.commit()
    return {
        "message": f"Generated questionnaires for {len(generated)} candidate(s)",
        "job_id": job_id,
        "top_n": top_n,
        "questions_per_candidate": questions_per_candidate,
        "candidates": generated,
    }


async def generate_questionnaires_by_topic_for_job(
    job_id: int,
    top_n: int,
    topic_counts: dict[str, int],
    db: sqlite3.Connection,
) -> dict:
    total_questions = sum(topic_counts.values())
    if total_questions <= 0:
        raise HTTPException(400, "At least one topic count must be greater than zero.")

    job_row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job_row:
        raise HTTPException(404, "Job not found")

    await ensure_all_candidates_ranked(job_id, db)

    top_rows = db.execute(
        """SELECT * FROM candidates
           WHERE job_id = ? AND rank > 0
           ORDER BY rank ASC, total_score DESC, semantic_score DESC
           LIMIT ?""",
        (job_id, top_n),
    ).fetchall()
    if not top_rows:
        raise HTTPException(404, "No ranked candidates available")

    cur = db.cursor()
    job = dict(job_row)
    questionnaire_semaphore = asyncio.Semaphore(QUESTIONNAIRE_CONCURRENCY)
    generated = await asyncio.gather(
        *[
            _build_topic_questionnaire_for_candidate(
                job,
                dict(candidate_row),
                topic_counts,
                questionnaire_semaphore,
            )
            for candidate_row in top_rows
        ]
    )

    for item in generated:
        cur.execute(
            "DELETE FROM candidate_questionnaires WHERE candidate_id = ?",
            (item["candidate_id"],),
        )
        for idx, question in enumerate(item["questions"], 1):
            cur.execute(
                """INSERT INTO candidate_questionnaires
                       (job_id, candidate_id, question_order, question_text, focus_area,
                        difficulty, reasoning, generation_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    item["candidate_id"],
                    idx,
                    question.get("question", "").strip(),
                    question.get("focus_area", "general"),
                    question.get("difficulty", "medium"),
                    question.get("reason", ""),
                    item["generation_mode"],
                ),
            )

    db.commit()
    return {
        "message": f"Generated topic-based questionnaires for {len(generated)} candidate(s)",
        "job_id": job_id,
        "top_n": top_n,
        "question_plan": topic_counts,
        "questions_per_candidate": total_questions,
        "candidates": generated,
    }


async def run_master_pipeline_for_job(
    job_id: int,
    top_n: int,
    questions_per_candidate: int,
    files: list[UploadFile] | None,
    db: sqlite3.Connection,
) -> dict:
    job_exists = db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job_exists:
        raise HTTPException(404, "Job not found")

    upload_result: dict | None = None
    uploaded_files = [f for f in (files or []) if f is not None]
    if uploaded_files:
        upload_result = await upload_resumes_for_job(job_id, uploaded_files, db)

    candidate_count = db.execute(
        "SELECT COUNT(*) FROM candidates WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]
    if candidate_count == 0:
        raise HTTPException(
            404,
            "No candidates found for this job. Upload resumes first or pass files to this endpoint.",
        )

    ranking_result = await rank_candidates_for_job(job_id, db)
    questionnaire_result = await generate_questionnaires_for_job(
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
