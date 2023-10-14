from pydantic import BaseModel
from typing import List

from surface_chat.serve.types import WorkerStatus


class RegisterWorkerRequest(BaseModel):
    worker_name: str
    check_heart_beat: bool
    worker_status: WorkerStatus


class WorkerAddressRequest(BaseModel):
    model: str


class WorkerAddressResponse(BaseModel):
    address: str


class ListModelsResponse(BaseModel):
    models: List[str]


class HeartbeatRequest(BaseModel):
    worker_name: str
    queue_length: int


class HeartbeatResponse(BaseModel):
    exist: bool
