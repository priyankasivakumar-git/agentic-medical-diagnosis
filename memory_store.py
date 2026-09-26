import json
import os
from datetime import datetime

class Memory:
    def __init__(self):
        self.history = []

    def log(self, agent_name, output):
        self.history.append({
            "agent": agent_name,
            "output": output,
        })

    def to_dict(self):
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_calls": self.history,
        }

    def save(self, output_dir="outputs/logs"):
        os.makedirs(output_dir, exist_ok=True)
        filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.json")
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return filepath