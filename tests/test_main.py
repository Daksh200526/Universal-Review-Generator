import asyncio
import tempfile
import unittest
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient

import main


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="My experience was helpful.")
                )
            ]
        )


class FormSubmitParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = {}

    def handle_starttag(self, tag, attrs):
        if tag != "form":
            return
        attributes = dict(attrs)
        form_id = attributes.get("id")
        if form_id in {"project-form", "survey-form"}:
            self.forms[form_id] = attributes


class ReviewGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_path = main.DATABASE_PATH
        self.original_client = main.client
        main.DATABASE_PATH = f"{self.temp_dir.name}\\test.sqlite3"
        main.initialize_database()
        main.client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

    def tearDown(self):
        main.DATABASE_PATH = self.original_database_path
        main.client = self.original_client
        self.temp_dir.cleanup()

    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def create_project(self, project_id, review_limit):
        request = main.ProjectRequest(
            id=project_id,
            title=project_id,
            redirectUrl="https://example.com/reviews",
            reviewLimit=review_limit,
            questions=[
                main.Question(
                    id="q1",
                    text="What did you experience?",
                    type="single",
                    options=["Helpful service"],
                )
            ],
        )
        return self.run_async(main.create_project(request))

    def generate(self, project_id, option="Helpful service"):
        request = main.ReviewRequest(
            project_id=project_id, selected_options=[option]
        )
        return self.run_async(main.generate_review(request))

    def test_limit_counts_regenerations_and_history_stays_with_its_project(self):
        self.create_project("limited", 1)
        self.create_project("unlimited", None)

        generated = self.generate("limited")
        self.assertEqual(generated["reviewCount"], 1)
        with self.assertRaises(HTTPException) as capped:
            self.generate("limited")
        self.assertEqual(capped.exception.status_code, 429)

        self.generate("unlimited")
        limited_history = self.run_async(
            main.list_project_reviews("limited")
        )
        unlimited_history = self.run_async(
            main.list_project_reviews("unlimited")
        )
        self.assertEqual(len(limited_history), 1)
        self.assertEqual(limited_history[0]["review"], "My experience was helpful.")
        self.assertEqual(len(unlimited_history), 1)

    def test_invalid_survey_option_does_not_consume_project_limit(self):
        self.create_project("validated", 1)

        with self.assertRaises(HTTPException) as invalid:
            self.generate("validated", option="Not a configured option")
        self.assertEqual(invalid.exception.status_code, 400)

        generated = self.generate("validated")
        self.assertEqual(generated["reviewCount"], 1)

    def test_limit_reopens_after_midnight_in_configured_timezone(self):
        self.create_project("daily", 1)
        start, end = main.review_day_bounds()
        with main.connect_database() as connection:
            connection.execute(
                """
                INSERT INTO reviews
                    (id, project_id, status, review_text, selected_options_json, created_at)
                VALUES (?, ?, 'completed', ?, ?, ?)
                """,
                (
                    "yesterday-review",
                    "daily",
                    "Old draft",
                    '["Helpful service"]',
                    start - 1,
                ),
            )

        boundary = datetime.fromtimestamp(start, main.REVIEW_LIMIT_TIMEZONE)
        self.assertEqual(main.review_day_bounds(boundary), (start, end))
        generated = self.generate("daily")

        self.assertEqual(generated["reviewCount"], 1)

    def test_project_survey_has_no_student_or_admin_project_mode(self):
        self.create_project("question-project", None)

        public_project = self.run_async(main.get_public_project("question-project"))
        self.assertIn("questions", public_project)
        self.assertNotIn("mode", public_project)
        self.assertNotIn("experts", public_project)

    def test_admin_project_api_is_available_without_sign_in(self):
        with TestClient(main.app) as client:
            response = client.get("/api/projects")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("/api/auth/login", {route.path for route in main.app.routes})

    def test_project_and_survey_forms_register_submit_handlers_once(self):
        html = (Path(main.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
        parser = FormSubmitParser()
        parser.feed(html)

        self.assertEqual(set(parser.forms), {"project-form", "survey-form"})
        self.assertNotIn("onsubmit", parser.forms["project-form"])
        self.assertNotIn("onsubmit", parser.forms["survey-form"])
        self.assertEqual(
            html.count("addEventListener('submit', handleSaveProject)"), 1
        )
        self.assertEqual(
            html.count("addEventListener('submit', handleGenerateReview)"), 1
        )


if __name__ == "__main__":
    unittest.main()
