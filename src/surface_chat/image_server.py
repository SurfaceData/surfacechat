import uvicorn

from argparse import ArgumentParser
from fastapi import FastAPI
from loguru import logger

from surface_chat.serve.routes import sdxl

app = FastAPI()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    app.include_router(sdxl.prepare_router())

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
