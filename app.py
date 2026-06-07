import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.config_routes import router as config_router
from api.stream_routes import router as stream_router
from api.task_routes import router as task_router
from services.runtime_manager import RuntimeManager


os.environ["PYMM_LOG_FILE"] = "0"
os.environ["BFIO_LOG_TO_FILE"] = "0"


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger("uvicorn.error")
logger.info("Server is starting. Please wait for 'Application startup complete.' before opening http://127.0.0.1:8000")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("microscope_api.log"),
        logging.StreamHandler(),
    ],
)

app = FastAPI(
    title="AI Microscope Assistant",
    description="AI-assisted microscope control and image analysis API service",
    version="2.0.0",
)
app.state.runtime_manager = RuntimeManager(ROOT_DIR)
app.include_router(config_router)
app.include_router(task_router)
app.include_router(stream_router)


@app.on_event("startup")
async def startup_event() -> None:
    await app.state.runtime_manager.startup()
    logger.info("Backend startup finished. The web UI is ready at http://127.0.0.1:8000")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.runtime_manager.release_system()


@app.get("/")
async def serve_frontend():
    index_path = ROOT_DIR / "front" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h2>front/index.html not found</h2>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
