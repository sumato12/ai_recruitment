import sqlite3

from fastapi import APIRouter, Depends

from database import get_db
from recruitment_api.workflows import rank_candidates_for_job

router = APIRouter(tags=["Ranking"])


@router.post("/jobs/{job_id}/rank-candidates/")
async def rank_candidates(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    return await rank_candidates_for_job(job_id, db)

