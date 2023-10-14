"""
A server wide tracker for all model workers.

A major refactor of FastChat's controller server:
    https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/controller.py
"""

import numpy as np
import requests
import time

from loguru import logger
from typing import List

from surface_chat.serve.types import DispatchMethod, WorkerInfo, WorkerStatus


class WorkerTracker:
    def __init__(self, dispatch_method: DispatchMethod, expiration_time: int = 90):
        self.worker_info = {}
        self.dispatch_method = dispatch_method
        self.expiration_time = expiration_time

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None
        return WorkerStatus(**r.json())

    def register_worker(
        self, worker_name: str, check_heart_beat: bool, worker_status: dict
    ):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
            logger.info(worker_status)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            model_names=worker_status.model_names,
            speed=worker_status.speed,
            queue_length=worker_status.queue_length,
            check_heart_beat=check_heart_beat,
            last_heart_beat=time.time(),
        )

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        if self.dispatch_method == DispatchMethod.lottery:
            return self.get_lottery_worker(model_name)
        elif self.dispatch_method == DispatchMethod.shortest_queue:
            return self.get_shortest_queue_worker(model_name)
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def get_lottery_worker(self, model_name: str):
        worker_names = []
        worker_speeds = []
        for w_name, w_info in self.worker_info.items():
            if model_name in w_info.model_names:
                worker_names.append(w_name)
                worker_speeds.append(w_info.speed)
        worker_speeds = np.array(worker_speeds, dtype=np.float32)
        norm = np.sum(worker_speeds)
        if norm < 1e-4:
            raise ValueError(f"No valid worker speeds")
        worker_speeds = worker_speeds / norm
        pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
        worker_name = worker_names[pt]
        return worker_name

    def get_shortest_queue_worker(self, model_name: str):
        worker_names = []
        worker_qlen = []
        for w_name, w_info in self.worker_info.items():
            if model_name in w_info.model_names:
                worker_names.append(w_name)
                worker_qlen.append(w_info.queue_length / w_info.speed)
        if len(worker_names) == 0:
            raise ValueError(f"No valid workers")
        min_index = np.argmin(worker_qlen)
        w_name = worker_names[min_index]
        self.worker_info[w_name].queue_length += 1
        logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
        return w_name

    def remove_stale_workers_by_expiration(self):
        expire = time.time() - self.expiration_time
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)
