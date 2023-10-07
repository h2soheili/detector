from typing import Any

from fastapi import APIRouter

from backend.global_log import structlog
from backend.instances import stream_manager, user
from backend.schemas import (StreamCreateDTO, StreamInDB)

logger = structlog.get_logger(__name__)

api_v1_router = APIRouter()


@api_v1_router.post('/stream', response_model=StreamInDB)
async def add_stream(data: StreamCreateDTO) -> Any:
    print(1)
    stream = StreamInDB(**data.model_dump(), user=user)
    # stream = StreamInDB(**data.model_dump(), id=1, user=user)
    stream_manager.add_stream(stream)
    return stream


@api_v1_router.delete('/stream/{id}', response_model=Any)
async def remove_stream(id: Any) -> Any:
    stream_manager.remove_stream(int(id))
