import matplotlib.pyplot as plt
from typing import List
import json
import argparse

import numpy as np

from pyskewadjuster import *

parser = argparse.ArgumentParser(
    description="JSON Trace Data to apply skew correction algorithm to"
)
parser.add_argument("file", type=str, action="store", help="Input file")

args = parser.parse_args()

data = None
with open(args.file, "r") as f:
    data = json.load(f)["data"]

services = discover_services(data)


def get_latency_distribution(tracedata, type="raw"):
    """Gets the latency distribution of a trace."""
    services = discover_services(tracedata)
    latency_distribution = {}
    for service in services:
        latency_distribution[service] = []

    for trace in tracedata:
        call_tree = generate_call_tree(trace)
        span_lookup = {span["spanID"]: span for span in trace["spans"]}
        bfs_queue = list(call_tree.keys())
        while bfs_queue:
            parent = span_lookup[bfs_queue.pop(0)]
            service = parent["operationName"]
            children = parent["childSpanIds"]
            for child in children:
                bfs_queue.append(child)
                child = span_lookup[child]
                delta = _get_latency(parent, child, type=type)
                latency_distribution[service].append(delta)

    for service in services:
        latency_distribution[service] = np.array(latency_distribution[service])

    return latency_distribution


def _get_latency(parentSpan, childSpan, type="raw"):
    """Gets the latency between a parent and child span."""
    try:
        if type == "raw":
            _, delta = get_ntp_params_pair(parentSpan, childSpan)
            return delta
        elif type == "forward":
            return int(get_attribute_from_tags(childSpan, "forward_latency_ns"))
        elif type == "backward":
            return int(get_attribute_from_tags(childSpan, "back_latency_ns"))
    except Exception as e:
        print(f"Error: {e}")
        print("Error: 'forward_latency_ns' or 'backward_latency_ns' not found in tags.")


def plot_latency_distribution(latency_distribution, fig=None, ax=None):
    """Plots the latency distribution of a trace."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    for service, latencies in latency_distribution.items():
        latencies = np.array(latencies) / 1e6
        ax.hist(latencies, bins=100, alpha=0.5, label=service)


latency_distribution_raw = get_latency_distribution(data, type="raw")
latency_distribution_f = get_latency_distribution(data, type="forward")
latency_distribution_b = get_latency_distribution(data, type="backward")

fig, ax = plt.subplots()
# plot_latency_distribution(latency_distribution_f, fig=fig, ax=ax)
# plot_latency_distribution(latency_distribution_b, fig=fig, ax=ax)
latency_asymmetry = {
    service: 0.5 * (latency_distribution_f[service] - latency_distribution_b[service])
    for service in services
}
plot_latency_distribution(latency_distribution_raw, fig=fig, ax=ax)

ax.set_xlabel("Latency (ms)")
fig.savefig("latency_distribution.png")
