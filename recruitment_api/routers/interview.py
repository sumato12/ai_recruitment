import sqlite3
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from database import get_db
from recruitment_api.schemas import TopicQuestionPlan
from recruitment_api.workflows import (
    generate_questionnaires_by_topic_for_job,
    generate_questionnaires_for_job,
    run_master_pipeline_for_job,
)

router = APIRouter(tags=["Interview"])


@router.post("/jobs/{job_id}/generate-questionnaires/")
async def generate_questionnaires(
    job_id: int,
    top_n: int = Query(5, ge=1, le=50),
    questions_per_candidate: int = Query(8, ge=3, le=20),
    db: sqlite3.Connection = Depends(get_db),
):
    return await generate_questionnaires_for_job(
        job_id=job_id,
        top_n=top_n,
        questions_per_candidate=questions_per_candidate,
        db=db,
    )


@router.post("/jobs/{job_id}/generate-questionnaires-by-topic/")
async def generate_questionnaires_by_topic(
    job_id: int,
    plan: TopicQuestionPlan,
    db: sqlite3.Connection = Depends(get_db),
):
    topic_counts = plan.counts()
    return await generate_questionnaires_by_topic_for_job(
        job_id=job_id,
        top_n=plan.top_n,
        topic_counts=topic_counts,
        db=db,
    )


@router.get("/jobs/{job_id}/questionnaires/")
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


@router.get("/candidates/{candidate_id}/questionnaire/")
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


@router.post(
    "/jobs/{job_id}/run-pipeline/",
    tags=["Run Full Pipeline"],
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
    return await run_master_pipeline_for_job(
        job_id=job_id,
        top_n=top_n,
        questions_per_candidate=questions_per_candidate,
        files=files,
        db=db,
    )

