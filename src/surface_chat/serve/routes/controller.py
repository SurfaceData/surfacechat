"""
API routes for a model worker controller.

A major refactor of FastChat's controller server:
    https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/controller.py
"""

import asyncio
import os
import time
import threading

from fastapi import APIRouter
from loguru import logger

from surface_chat.serve.controller_types import (
    HeartbeatRequest,
    HeartbeatResponse,
    ListModelsResponse,
    RegisterWorkerRequest,
    WorkerAddressRequest,
    WorkerAddressResponse,
)
from surface_chat.serve.worker_tracker import WorkerTracker


# Move to settings
CONTROLLER_HEART_BEAT_EXPIRATION = int(
    os.getenv("FASTCHAT_CONTROLLER_HEART_BEAT_EXPIRATION", 90)
)


router = APIRouter(
    prefix="/v1/controller",
    tags=["controller"],
)


@router.post("/register_worker")
def register_worker(request: RegisterWorkerRequest):
    router.worker_tracker.register_worker(
        request.worker_name, request.check_heart_beat, request.worker_status
    )


@router.post("/refresh_all_workers")
async def refresh_all_workers():
    router.worker_tracker.refresh_all_workers()


@router.post("/list_models")
async def list_models():
    return ListModelsResponse(models=router.worker_tracker.list_models())


@router.post("/get_worker_address")
async def get_worker_address(request: WorkerAddressRequest):
    return WorkerAddressResponse(
        address=router.worker_tracker.get_worker_address(request.model)
    )


@router.post("/receive_heart_beat")
async def receive_heart_beat(request: HeartbeatRequest):
    worker_name = request.worker_name
    queue_length = request.queue_length
    worker_info = router.worker_tracker.worker_info
    if request.worker_name not in worker_info:
        logger.info(f"Receive unknown heart beat. {worker_name}")
        return HeartbeatResponse(exist=False)

    worker_info[worker_name].queue_length = queue_length
    worker_info[worker_name].last_heart_beat = time.time()
    return HeartbeatResponse(exist=True)


class WorkerCleaner:
    async def run_main(self):
        while True:
            logger.info("Cleaning")
            await asyncio.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
            router.worker_tracker.remove_stale_workers_by_expiration()


runner = WorkerCleaner()


@router.on_event("startup")
async def app_startup():
    asyncio.create_task(runner.run_main())


def prepare_router(worker_tracker: WorkerTracker):
    router.worker_tracker = worker_tracker
    return router
