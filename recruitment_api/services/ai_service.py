import json
import re
import time
import asyncio
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI, OpenAI

from config import (
    AI_MAX_RETRIES,
    AI_RETRY_BACKOFF_SECONDS,
    AI_TIMEOUT_SECONDS,
    EMBEDDING_MODEL_NAME,
    LLM_SCORE_WEIGHT,
    SEMANTIC_SCORE_WEIGHT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_MODEL,
)

client = OpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)
async_client = AsyncOpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)


def _clean_json(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_embedding_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers.utils.import_utils as hf_import_utils
        hf_import_utils._torchvision_available = False
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Embedding dependencies missing. Install with: pip install transformers torch"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        model.eval()
        return tokenizer, model, torch
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {exc}"
        ) from exc


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_job_embedding_text(
    title: str,
    description: str,
    skills: Any | None = None,
    experience: Any | None = None,
) -> str:
    parts = [
        f"Job Title: {_clean_text(title)}",
        f"Job Description:\n{_clean_text(description)}",
    ]

    skills_text = _clean_text(skills)
    if skills_text:
        parts.append(f"Required Skills: {skills_text}")

    experience_text = _clean_text(experience)
    if experience_text:
        parts.append(f"Minimum Experience: {experience_text} years")

    return "\n".join(parts).strip()


def build_candidate_embedding_text(candidate: dict) -> str:
    parts = [
        f"Name: {_clean_text(candidate.get('name'))}",
        f"Experience: {_clean_text(candidate.get('total_experience'))}",
        f"Skills: {_clean_text(candidate.get('skills'))}",
        f"Education: {_clean_text(candidate.get('education'))}",
        f"Certifications: {_clean_text(candidate.get('certifications'))}",
        f"Summary: {_clean_text(candidate.get('summary'))}",
    ]

    raw_text = _clean_text(candidate.get("raw_text"))
    if raw_text:
        parts.append(f"Resume Text:\n{raw_text[:12000]}")

    return "\n".join(p for p in parts if p.strip()).strip()


def get_embedding(text: str, *, is_query: bool) -> list[float]:
    clean = _clean_text(text)
    if not clean:
        return []

    prefix = "query: " if is_query else "passage: "
    tokenizer, model, torch = _get_embedding_stack()
    encoded = tokenizer(
        prefix + clean,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return normalized[0].cpu().tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot


def semantic_score(vec_a: list[float], vec_b: list[float]) -> float:
    return max(0.0, min(100.0, cosine_similarity(vec_a, vec_b) * 100))


def combine_scores(semantic: float, llm: float | None = None) -> float:
    if llm is None:
        return round(semantic, 2)

    total_weight = SEMANTIC_SCORE_WEIGHT + LLM_SCORE_WEIGHT
    if total_weight <= 0:
        return round(semantic, 2)

    blended = (
        (semantic * SEMANTIC_SCORE_WEIGHT) + (llm * LLM_SCORE_WEIGHT)
    ) / total_weight
    return round(blended, 2)


def _chat_completion_json(system_prompt: str, user_prompt: str) -> dict:
    """Call AI with timeout/retry and parse JSON output."""
    last_error = None

    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                timeout=AI_TIMEOUT_SECONDS,
            )
            content = resp.choices[0].message.content or ""
            return json.loads(_clean_json(content))
        except Exception as exc:
            last_error = exc
            if attempt == AI_MAX_RETRIES:
                break
            time.sleep(AI_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"AI request failed after {AI_MAX_RETRIES} attempt(s): {last_error}"
    ) from last_error


async def _chat_completion_json_async(system_prompt: str, user_prompt: str) -> dict:
    """Async AI call with timeout/retry and JSON parsing."""
    last_error = None

    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            resp = await async_client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                timeout=AI_TIMEOUT_SECONDS,
            )
            content = resp.choices[0].message.content or ""
            return json.loads(_clean_json(content))
        except Exception as exc:
            last_error = exc
            if attempt == AI_MAX_RETRIES:
                break
            await asyncio.sleep(AI_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"AI request failed after {AI_MAX_RETRIES} attempt(s): {last_error}"
    ) from last_error


def extract_candidate_info(resume_text: str) -> dict:
    """Use GPT to pull structured fields from raw resume text."""

    prompt = f"""Extract the following information from this resume.
Return a JSON object with exactly these keys:
- "name"              : string (full name)
- "email"             : string or null
- "phone"             : string or null
- "total_experience"  : string (e.g. "5 years"); estimate from work history if not stated
- "skills"            : string, comma-separated list of key skills
- "education"         : string, comma-separated degrees / institutions
- "certifications"    : string, comma-separated or empty string
- "summary"           : string, 2-3 sentence professional summary

Resume:
\"\"\"
{resume_text[:12000]}
\"\"\"

Return ONLY the JSON object. No markdown, no explanation."""

    return _chat_completion_json(
        system_prompt="You are a resume-parsing assistant. Return only valid JSON.",
        user_prompt=prompt,
    )


async def extract_candidate_info_async(resume_text: str) -> dict:
    """Async version: use GPT to pull structured fields from raw resume text."""

    prompt = f"""Extract the following information from this resume.
Return a JSON object with exactly these keys:
- "name"              : string (full name)
- "email"             : string or null
- "phone"             : string or null
- "total_experience"  : string (e.g. "5 years"); estimate from work history if not stated
- "skills"            : string, comma-separated list of key skills
- "education"         : string, comma-separated degrees / institutions
- "certifications"    : string, comma-separated or empty string
- "summary"           : string, 2-3 sentence professional summary

Resume:
\"\"\"
{resume_text[:12000]}
\"\"\"

Return ONLY the JSON object. No markdown, no explanation."""

    return await _chat_completion_json_async(
        system_prompt="You are a resume-parsing assistant. Return only valid JSON.",
        user_prompt=prompt,
    )


def score_candidate(
    job_description: str,
    job_skills: str,
    min_experience: int | str,
    candidate: dict,
) -> dict:
    """Score a candidate 0-100 against a job description."""

    prompt = f"""You are an expert HR recruiter. Score this candidate against the full job requirement set (0-100).

Weighting:
  Skill match          – 40 %
  Experience relevance – 30 %
  Education / certs    – 20 %
  Overall fit          – 10 %

Important:
- Compare candidate skills against required job skills.
- Compare candidate total experience against minimum required experience.
- If required skills or minimum experience are clearly missing, reduce the score accordingly.
- Mention key skill and experience gaps in reasoning when present.

Job Description:
\"\"\"
{job_description[:6000]}
\"\"\"
Required Skills: {job_skills or "N/A"}
Minimum Experience (years): {min_experience}

Candidate:
  Name           : {candidate.get("name", "N/A")}
  Experience     : {candidate.get("total_experience", "N/A")}
  Skills         : {candidate.get("skills", "N/A")}
  Education      : {candidate.get("education", "N/A")}
  Certifications : {candidate.get("certifications", "N/A")}
  Summary        : {candidate.get("summary", "N/A")}

Return ONLY a JSON object:
  {{"score": <int 0-100>, "reasoning": "<2-3 sentences>"}}"""

    return _chat_completion_json(
        system_prompt="You are an HR scoring assistant. Return only valid JSON.",
        user_prompt=prompt,
    )


async def score_candidate_async(
    job_description: str,
    job_skills: str,
    min_experience: int | str,
    candidate: dict,
) -> dict:
    """Async version: score a candidate 0-100 against a job description."""

    prompt = f"""You are an expert HR recruiter. Score this candidate against the full job requirement set (0-100).

Weighting:
  Skill match          – 40 %
  Experience relevance – 30 %
  Education / certs    – 20 %
  Overall fit          – 10 %

Important:
- Compare candidate skills against required job skills.
- Compare candidate total experience against minimum required experience.
- If required skills or minimum experience are clearly missing, reduce the score accordingly.
- Mention key skill and experience gaps in reasoning when present.

Job Description:
\"\"\"
{job_description[:6000]}
\"\"\"
Required Skills: {job_skills or "N/A"}
Minimum Experience (years): {min_experience}

Candidate:
  Name           : {candidate.get("name", "N/A")}
  Experience     : {candidate.get("total_experience", "N/A")}
  Skills         : {candidate.get("skills", "N/A")}
  Education      : {candidate.get("education", "N/A")}
  Certifications : {candidate.get("certifications", "N/A")}
  Summary        : {candidate.get("summary", "N/A")}

Return ONLY a JSON object:
  {{"score": <int 0-100>, "reasoning": "<2-3 sentences>"}}"""

    return await _chat_completion_json_async(
        system_prompt="You are an HR scoring assistant. Return only valid JSON.",
        user_prompt=prompt,
    )
