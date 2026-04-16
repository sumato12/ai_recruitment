import sqlite3
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from database import get_db
from recruitment_api.workflows import upload_resumes_for_job

router = APIRouter(tags=["Resumes"])


@router.post("/jobs/{job_id}/upload-resumes/")
async def upload_resumes(
    job_id: int,
    files: List[UploadFile] = File(...),
    db: sqlite3.Connection = Depends(get_db),
):
    return await upload_resumes_for_job(job_id, files, db)

