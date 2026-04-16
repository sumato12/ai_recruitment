import os

from fastapi import FastAPI

from config import EXPORT_DIR, UPLOAD_DIR
from database import create_tables
from recruitment_api.routers import candidates, export, interview, jobs, ranking, resumes

app = FastAPI(title="AI Recruitment Module", version="1.0.0")


@app.on_event("startup")
def startup():
    create_tables()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)


app.include_router(jobs.router)
app.include_router(resumes.router)
app.include_router(ranking.router)
app.include_router(interview.router)
app.include_router(candidates.router)
app.include_router(export.router)

