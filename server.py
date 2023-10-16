import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from ai.my_processes import serve_model, send_stream_to_model_process
from backend.api import api_v1_router
from backend.config import settings
from backend.global_log import structlog
from backend.instances import (socket_manager, loop, model_process_in_queue, model_process_out_queue,
                               shared_data_for_stream_config, detector, stream, stream_manager,
                               stream_process_out_queue)

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="model_server",
    version="1.0.0",
    description="description",
    openapi_url="/openapi.json",
    debug=True,
    docs_url="/docs",
    redoc_url="/recode",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

swagger_js_url = "/static/swagger-ui-bundle.js"
swagger_css_url = "/static/swagger-ui.css"


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url=swagger_js_url,
        swagger_css_url=swagger_css_url,
    )


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
async def running():
    return f"Server is running  ...  see /docs for more info"


@app.get("/health", response_model=Any)
async def get_state():
    return "log to .."


@app.get('/server-config', response_model=Any)
async def get_server_config() -> Any:
    return settings.model_dump(mode='json')


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await socket_manager.connect(websocket)
    except Exception as e:
        print(e)
    try:
        while True:
            data = await websocket.receive_text()
            socket_manager.on_message(data)
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket)


api_router = APIRouter()

api_router.include_router(api_v1_router, tags=["v1"])
app.include_router(api_v1_router, prefix="/v1")


@app.on_event("startup")
async def startup_event():
    """
    lazy load model to not locking app start time
    """
    await asyncio.sleep(1)
    pool1 = ThreadPoolExecutor(max_workers=2)
    # pool2 = ThreadPoolExecutor(max_workers=2)
    loop.run_in_executor(pool1, serve_model, model_process_in_queue, model_process_out_queue, detector,
                         shared_data_for_stream_config)
    # loop.run_in_executor(pool2, send_stream_to_model_process, model_process_in_queue, stream_process_out_queue)
    """
    auto add stream for development purpose
    """
    stream_manager.add_stream(stream)



@app.on_event("shutdown")
async def shutdown_event():
    await asyncio.sleep(0.01)
