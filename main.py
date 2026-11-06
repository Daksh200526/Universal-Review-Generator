import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Universal Review Generator")

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/static/{file_path:path}")
async def serve_static_file(file_path: str):
    static_root = os.path.realpath(STATIC_DIR)
    requested_file = os.path.realpath(os.path.join(static_root, file_path))
    if os.path.commonpath([static_root, requested_file]) != static_root or not os.path.isfile(requested_file):
        raise HTTPException(status_code=404, detail="Static file not found.")
    return FileResponse(requested_file)


class ReviewRequest(BaseModel):
    business_name: str
    mode: str = "customer"  # "customer" or "student"
    selected_options: List[str]
    expert_name: Optional[str] = None


@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "index.html not found in static folder."}


@app.post("/api/generate-review")
async def generate_review(req: ReviewRequest):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server."
        )

    if req.mode == "student":
        system_prompt = (
            "You are a real student or candidate writing an authentic, heartfelt recommendation "
            "for mentorship/counseling. Your tone must be personal, relatable, storytelling-driven, "
            "and sound like one student giving trusted peer-to-peer advice to another student who was "
            "in the exact same confused or uncertain situation. Avoid sounding like a formal corporate advertisement. "
            "Output ONLY the review text with no quotes, headings, or introductory pleasantries."
        )

        user_prompt = f"""
Platform / Institution / Service: {req.business_name}
Mentor / Expert Consulted: {req.expert_name or 'the counselor'}
Specific expertise and aspects experienced:
{", ".join(req.selected_options)}

Write a concise 3 to 5 sentence personal testimonial. Start with where the student was initially feeling stuck or looking for direction, how consulting {req.expert_name or 'the expert'} made everything clear, and directly urge other students in the same boat to reach out to them.
"""
    else:
        # Standard customer mode
        system_prompt = (
            "You are an authentic, satisfied customer writing a genuine, positive public review "
            "for a local business. Write natural, conversational sentences without sounding robotic, "
            "repetitive, or like an advertisement billboard. Output ONLY the review text with no quotes or intros."
        )

        user_prompt = f"""
Business Name: {req.business_name}
Specific positive details I experienced:
{", ".join(req.selected_options)}

Write a concise 2 to 4 sentence 5-star review smoothly weaving in these details.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.88,
            max_tokens=260
        )
        review_text = response.choices[0].message.content.strip().strip('"')
        return {"review": review_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}