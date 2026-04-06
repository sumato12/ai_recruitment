import os


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://kdpbookgeneration.openai.azure.com/openai/v1",
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

DATABASE_PATH = "recruitment.db"
UPLOAD_DIR = "uploads"
EXPORT_DIR = "exports"

UPLOAD_MAX_SIZE_MB = int(os.getenv("UPLOAD_MAX_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = UPLOAD_MAX_SIZE_MB * 1024 * 1024

AI_TIMEOUT_SECONDS = float(os.getenv("AI_TIMEOUT_SECONDS", "45"))
AI_MAX_RETRIES = max(1, int(os.getenv("AI_MAX_RETRIES", "3")))
AI_RETRY_BACKOFF_SECONDS = float(os.getenv("AI_RETRY_BACKOFF_SECONDS", "1.0"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-base-v2")
USE_LLM_IN_RANKING = _get_bool("USE_LLM_IN_RANKING", True)
SEMANTIC_SCORE_WEIGHT = float(os.getenv("SEMANTIC_SCORE_WEIGHT", "0.7"))
LLM_SCORE_WEIGHT = float(os.getenv("LLM_SCORE_WEIGHT", "0.3"))
SKILL_SCORE_WEIGHT = float(os.getenv("SKILL_SCORE_WEIGHT", "0.4"))
PROJECTS_SCORE_WEIGHT = float(os.getenv("PROJECTS_SCORE_WEIGHT", "0.3"))
EXPERIENCE_SCORE_WEIGHT = float(os.getenv("EXPERIENCE_SCORE_WEIGHT", "0.3"))
