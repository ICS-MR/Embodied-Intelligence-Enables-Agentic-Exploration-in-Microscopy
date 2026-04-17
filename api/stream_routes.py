from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_runtime_manager
from api.models import PreviewStatusResponse, UserInputResponse


router = APIRouter()


class UserInputRequest(BaseModel):
    text: str


@router.get("/api/stream/global")
async def global_message_stream(runtime_manager=Depends(get_runtime_manager)) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        async for item in runtime_manager.global_message_stream():
            yield item

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/stream/user_input", response_model=UserInputResponse)
async def receive_user_input(req: UserInputRequest, runtime_manager=Depends(get_runtime_manager)) -> UserInputResponse:
    return UserInputResponse.model_validate(await runtime_manager.receive_user_input(req.text))


@router.get("/video/stream")
async def video_stream(runtime_manager=Depends(get_runtime_manager)):
    return StreamingResponse(
        runtime_manager.generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/stream/preview_status", response_model=PreviewStatusResponse)
async def preview_status(runtime_manager=Depends(get_runtime_manager)) -> PreviewStatusResponse:
    return PreviewStatusResponse.model_validate(runtime_manager.get_preview_status())


@router.get("/api/artifacts/{artifact_path:path}")
async def get_runtime_artifact(artifact_path: str, runtime_manager=Depends(get_runtime_manager)):
    try:
        artifact = runtime_manager.resolve_runtime_artifact_path(artifact_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(artifact)
