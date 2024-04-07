from typing import List
import json

from .datastructs import Trace


def save_traces(traces: List[Trace]):
    """Save traces to a file, loadable by Jaeger."""
    out = {"data": [traces[i].__dict__ for i in range(len(traces))]}
    with open("trace.json", "w") as f:
        json.dump(out, f)
