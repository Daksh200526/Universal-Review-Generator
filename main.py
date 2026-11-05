import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Universal Review Generator")

# Path to the directory containing static assets
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Mount static folder for auxiliary assets (CSS, JS, images)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "index.html not found in static folder."}


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}