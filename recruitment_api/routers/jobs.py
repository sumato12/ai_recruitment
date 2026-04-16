import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from database import get_db
from recruitment_api.schemas import JobCreate
from recruitment_api.workflows import embedding_to_db
from services.ai_service import build_job_embedding_text, get_embedding

router = APIRouter(tags=["Jobs"])
logger = logging.getLogger(__name__)


@router.post("/jobs/")
def create_job(body: JobCreate, db: sqlite3.Connection = Depends(get_db)):
    embedding_json = None
    try:
        embedding_text = build_job_embedding_text(
            body.title, body.description, body.skills, body.experience
        )
        embedding_json = embedding_to_db(get_embedding(embedding_text, is_query=True))
    except Exception as exc:
        logger.warning("Job embedding generation failed for title '%s': %s", body.title, exc)
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


@router.get("/jobs/")
def list_jobs(db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return dict(row)


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.cursor()
    cur.execute("DELETE FROM candidate_questionnaires WHERE job_id = ?", (job_id,))
    cur.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
    cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Job not found")
    return {"message": "Job and its candidates deleted"}

