import json
import os
import time


class Tracer:
    def __init__(self, trace_dir: str):
        os.makedirs(trace_dir, exist_ok=True)
        self.trace_path = os.path.join(trace_dir, "agent_trace.jsonl")

    def log(self, record: dict) -> None:
        record["timestamp"] = time.time()
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
