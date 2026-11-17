import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="Universal Review Generator")
logger = logging.getLogger(__name__)

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "reviews.sqlite3"))
REVIEW_LIMIT_TIMEZONE_NAME = os.environ.get(
    "REVIEW_LIMIT_TIMEZONE", "Asia/Kolkata"
)
try:
    REVIEW_LIMIT_TIMEZONE = ZoneInfo(REVIEW_LIMIT_TIMEZONE_NAME)
except ZoneInfoNotFoundError as error:
    raise RuntimeError(
        f"Unknown REVIEW_LIMIT_TIMEZONE: {REVIEW_LIMIT_TIMEZONE_NAME}"
    ) from error


@contextmanager
def connect_database():
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with connect_database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                redirect_url TEXT NOT NULL,
                review_limit INTEGER,
                questions_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                review_text TEXT NOT NULL DEFAULT '',
                selected_options_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reviews_project_created_status
                ON reviews(project_id, created_at, status);
            """
        )


initialize_database()


class Question(BaseModel):
    id: str
    text: str
    type: Literal["single", "multiple"]
    options: List[str] = Field(min_length=1)


class ProjectRequest(BaseModel):
    id: Optional[str] = None
    title: str = Field(min_length=1, max_length=200)
    redirectUrl: HttpUrl
    reviewLimit: Optional[int] = Field(default=None, ge=1)
    questions: List[Question] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    project_id: str
    selected_options: List[str] = Field(min_length=1)


def project_record(row, include_usage=False):
    record = {
        "id": row["id"],
        "title": row["title"],
        "redirectUrl": row["redirect_url"],
        "reviewLimit": row["review_limit"],
        "questions": json.loads(row["questions_json"]),
    }
    if include_usage:
        record["reviewCount"] = row["review_count"]
        record["reviewLimitTimezone"] = REVIEW_LIMIT_TIMEZONE_NAME
        _, record["reviewLimitResetAt"] = review_day_bounds()
    return record


def review_day_bounds(now=None):
    if now is None:
        now = datetime.now(REVIEW_LIMIT_TIMEZONE)
    local_now = now.astimezone(REVIEW_LIMIT_TIMEZONE)
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def get_project_with_count(connection, project_id, now=None):
    start, end = review_day_bounds(now)
    return connection.execute(
        """
        SELECT p.*,
               (SELECT COUNT(*) FROM reviews r
                WHERE r.project_id = p.id
                  AND r.status IN ('pending', 'completed')
                  AND r.created_at >= ? AND r.created_at < ?)
                   AS review_count
        FROM projects p WHERE p.id = ?
        """,
        (start, end, project_id),
    ).fetchone()


@app.get("/static/{file_path:path}")
async def serve_static_file(file_path: str):
    static_root = os.path.realpath(STATIC_DIR)
    requested_file = os.path.realpath(os.path.join(static_root, file_path))
    if (
        os.path.commonpath([static_root, requested_file]) != static_root
        or not os.path.isfile(requested_file)
    ):
        raise HTTPException(status_code=404, detail="Static file not found.")
    return FileResponse(requested_file)


@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "index.html not found in static folder."}


@app.get("/api/projects")
async def list_projects():
    start, end = review_day_bounds()
    with connect_database() as connection:
        rows = connection.execute(
            """
            SELECT p.*,
                   (SELECT COUNT(*) FROM reviews r
                    WHERE r.project_id = p.id AND r.status = 'completed'
                      AND r.created_at >= ? AND r.created_at < ?) AS review_count
            FROM projects p ORDER BY p.created_at DESC
            """,
            (start, end),
        ).fetchall()
    return [project_record(row, include_usage=True) for row in rows]


@app.post("/api/projects")
async def create_project(req: ProjectRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Project title cannot be empty.")
    if not req.questions:
        raise HTTPException(status_code=422, detail="Add at least one survey question.")

    project_id = req.id or f"proj_{uuid.uuid4().hex}"
    with connect_database() as connection:
        connection.execute(
            """
            INSERT INTO projects
                (id, title, redirect_url, review_limit, questions_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                title,
                str(req.redirectUrl),
                req.reviewLimit,
                json.dumps([question.model_dump() for question in req.questions]),
                int(time.time()),
            ),
        )
        row = get_project_with_count(connection, project_id)
    return project_record(row, include_usage=True)


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    with connect_database() as connection:
        cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Project not found.")
    return {"deleted": True}


@app.get("/api/projects/{project_id}/reviews")
async def list_project_reviews(project_id: str):
    with connect_database() as connection:
        if not connection.execute(
            "SELECT 1 FROM projects WHERE id = ?", (project_id,)
        ).fetchone():
            raise HTTPException(status_code=404, detail="Project not found.")
        rows = connection.execute(
            """
            SELECT id, review_text, selected_options_json, created_at
            FROM reviews WHERE project_id = ? AND status = 'completed'
            ORDER BY created_at DESC
            """,
            (project_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "review": row["review_text"],
            "selectedOptions": json.loads(row["selected_options_json"]),
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


@app.get("/api/public/projects/{project_id}")
async def get_public_project(project_id: str):
    with connect_database() as connection:
        row = get_project_with_count(connection, project_id)
    if not row:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project_record(row, include_usage=True)


@app.post("/api/generate-review")
async def generate_review(req: ReviewRequest):
    if not client:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    review_id = f"review_{uuid.uuid4().hex}"
    with connect_database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        project = get_project_with_count(connection, req.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")

        questions = json.loads(project["questions_json"])
        valid_options = {
            option
            for question in questions
            for option in question["options"]
        }

        if any(option not in valid_options for option in req.selected_options):
            raise HTTPException(
                status_code=400, detail="Select only options from this project."
            )
        review_limit = project["review_limit"]
        if review_limit is not None and project["review_count"] >= review_limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    "This project has reached its daily review-generation limit. "
                    f"It resets at midnight {REVIEW_LIMIT_TIMEZONE_NAME}."
                ),
            )

        connection.execute(
            """
            INSERT INTO reviews
                (id, project_id, status, selected_options_json, created_at)
            VALUES (?, ?, 'pending', ?, ?)
            """,
            (
                review_id,
                req.project_id,
                json.dumps(req.selected_options),
                int(time.time()),
            ),
        )

    system_prompt = (
        "Help someone express their own genuine experience in a clear, natural way. "
        "Use only the details they selected; do not invent facts, make comparative claims, "
        "or add a rating. Return only a concise first-person draft."
    )
    user_prompt = (
        f"Organization: {project['title']}\n"
        f"Details from the person's experience: {', '.join(req.selected_options)}\n"
        "Write a concise first-person draft that reflects these details."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=260,
        )
        review_text = (response.choices[0].message.content or "").strip().strip('"')
        if not review_text:
            raise RuntimeError("The review provider returned an empty response.")
        with connect_database() as connection:
            connection.execute(
                "UPDATE reviews SET status = 'completed', review_text = ? WHERE id = ?",
                (review_text, review_id),
            )
            updated_project = get_project_with_count(connection, req.project_id)
        return {
            "review": review_text,
            "reviewCount": updated_project["review_count"],
        }
    except Exception as error:
        with connect_database() as connection:
            connection.execute(
                "DELETE FROM reviews WHERE id = ? AND status = 'pending'",
                (review_id,),
            )
        logger.exception("Review generation failed for project %s", req.project_id)
        raise HTTPException(status_code=502, detail="Review generation failed.") from error


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
