"""Exclusive synchronous pipeline timing, including nested VLM requests."""
from collections import defaultdict
from contextvars import ContextVar
from time import perf_counter

active_timer = ContextVar("hgr_stage_timer", default=None)


class StageTimer:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.stage = "other"
        self.started = self.last = perf_counter()
        self.stack = []

    def switch(self, stage):
        now = perf_counter()
        self.seconds[self.stage] += now - self.last
        self.last, self.stage = now, stage

    def push(self, stage):
        self.stack.append(self.stage)
        self.switch(stage)

    def pop(self):
        self.switch(self.stack.pop() if self.stack else "other")

    def snapshot(self):
        self.switch(self.stage)
        return {"stages_seconds": dict(self.seconds), "total_seconds": self.last-self.started}
