"""
A combined uvicorn API server that blends multiple API sets:
    - Model Worker Controller
    - OpenAI shim

This is a refactored mix of two FastChat servers:
    - https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/controller.py
    - https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/openai_api_server.py
"""

import uvicorn

from argparse import ArgumentParser
from fastapi import FastAPI
from loguru import logger

from surface_chat.serve.routes import controller, openai
from surface_chat.serve.types import DispatchMethod
from surface_chat.serve.worker_tracker import WorkerTracker

app = FastAPI()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument(
        "--dispatch-method",
        type=DispatchMethod,
        default=DispatchMethod.shortest_queue,
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker_tracker = WorkerTracker(args.dispatch_method)

    app.include_router(controller.prepare_router(worker_tracker))
    app.include_router(openai.prepare_router(worker_tracker))

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
