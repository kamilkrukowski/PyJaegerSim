from typing import List
import json
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from pyskewadjuster import *

parser = argparse.ArgumentParser(
    description="JSON Trace Data to apply skew correction algorithm to"
)
parser.add_argument("file", type=str, action="store", help="Input file")
parser.add_argument("--anchor", type=str, action="store", help="Anchor node, if not Random")
parser.add_argument("--final", action="store_true", help="Final iteration of the algorithm")


args = parser.parse_args()

data = None
with open(args.file, "r") as f:
    data = json.load(f)["data"]

services = discover_services(data)


def calculate_pairwise_skew_stats(tracedata, type="raw"):
    """Gets the latency distribution of a trace."""
    services = discover_services(tracedata)
    skew_corrections = dict()
    global_skews = dict()
    correction_error = dict()
    seen_services = set()

    for trace in tracedata:
        for span in trace["spans"]:
            operationName = span["operationName"]
            if operationName not in seen_services:

                correction = get_attribute_from_tags(span, "clock_skew_correction")
                correction = int(correction) if correction is not None else 0
                global_skew = int(get_attribute_from_tags(span, "global_skew_ns"))
                error = global_skew + correction

                seen_services.add(operationName)

                skew_corrections[operationName] = correction
                global_skews[operationName] = global_skew
                correction_error[operationName] = error

    pairwise_skew = dict()
    pairwise_error = dict()
    for service in services:
        pairwise_skew[service] = dict()
        pairwise_error[service] = dict()

    for service1, service2 in itertools.product(services, services):
        if service1 == service2:
            continue
        pairwise_skew[service1][service2] = (
            global_skews[service1] - global_skews[service2]
        )
        pairwise_error[service1][service2] = (
            correction_error[service1] - correction_error[service2]
        )

    return skew_corrections, pairwise_skew, pairwise_error


def plot_latency_distribution(latency_distribution, fig=None, ax=None):
    """Plots the latency distribution of a trace."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    for service, latencies in latency_distribution.items():
        for service2, latencies2 in latencies.items():
            if service == service2:
                continue
            latencies = latencies2
            latencies = np.array(latencies) / 1e6
            ax.hist(latencies, bins=100, alpha=0.5, label=service)


def calculate_heatmap(pairwise_skew):
    """Plots the pairwise skew of a trace."""
    heatmap = np.zeros((len(pairwise_skew), len(pairwise_skew)))
    services = list(pairwise_skew.keys())
    services = sorted(services)
    for service1, service2 in itertools.product(services, services):
        if service1 == service2:
            continue
        heatmap[services.index(service1)][services.index(service2)] = (
            pairwise_skew[service1][service2] / 1.0e6
        )
    return heatmap, services


import copy

n_iterations = len([1 for _ in correct_skew(copy.deepcopy(data), anchor=args.anchor, verbose=False)]) + 1 if not args.final else 2

skew_correction, pairwise_skew, pairwise_error = calculate_pairwise_skew_stats(
    data, type="raw"
)
fig, ax = plt.subplots(1, n_iterations, figsize=(n_iterations * 5, 6))
vmax = (
    np.max(
        [
            pairwise_skew[service].get(service2, 0)
            for service, service2 in itertools.product(services, services)
        ]
    )
    / 1.0e6
)
vmin = -vmax
cbar_ax = fig.add_axes([0.895, 0.2, 0.03, 0.5])
_cmap = "vlag"

def plot(skews, ax):
    heatmap, ticks = calculate_heatmap(skews)
    mask = ~np.tril(np.ones_like(heatmap, dtype=bool))
    identity = np.eye(len(heatmap), dtype=bool)
    heatmap[~(mask + identity)] -= heatmap[~(mask + identity)].mean()
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".2f",
        cmap=_cmap,
        cbar=False,
        ax=ax,
        vmax=vmax,
        vmin=vmin,
        mask=mask,
        xticklabels=ticks,
        yticklabels=ticks,
    )
    print(f"Mean error: ", np.abs(heatmap[mask]).mean())
    ax.set_xlabel("Latency (ms)")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Parent Node ID")
    ax.set_ylabel("Child Node ID")


heatmap1, ticks = calculate_heatmap(pairwise_skew)
mask = ~np.tril(np.ones_like(heatmap1, dtype=bool))
identity = np.eye(len(heatmap1), dtype=bool)
#heatmap1[~(mask + identity)] -= heatmap1[~(mask + identity)].mean()
print(f"Mean error: ", np.abs(heatmap1[mask]).mean())
sns.heatmap(
    heatmap1,
    annot=True,
    fmt=".2f",
    cmap=_cmap,
    cbar=True,
    cbar_ax=cbar_ax,
    ax=ax[0],
    vmax=vmax,
    vmin=vmin,
    mask=mask,
    xticklabels=ticks,
    yticklabels=ticks,
)
ax[0].set_xlabel("Latency (ms)")
ax[0].set_aspect("equal", "box")


fig.suptitle("Pairwise Clock Skew Error Before and After Protocol (ms)")

ax[0].set_title("Before Protocol")
ax[1].set_title("After Protocol")

ax[0].set_xlabel("Parent Node ID")
ax[0].set_ylabel("Child Node ID")

if args.final:
    generator = correct_skew(data, anchor=args.anchor, verbose=False)
    data = None
    while True:
        try:
            data = next(generator)
        except StopIteration as e:
            break
    skew_correction, pairwise_skew, pairwise_error = calculate_pairwise_skew_stats(
        data, type="corrected"
    )
    plot(pairwise_error, ax[1])
    fig.tight_layout(rect=[0.05, 0, 0.9, 0.95])
    fig.savefig("latency_hist_final.png")
    exit()
else:
    generator = correct_skew(data, anchor=args.anchor, verbose=True)
    data = next(generator)
    try:
        for i in range(1, n_iterations + 1):
            skew_correction, pairwise_skew, pairwise_error = calculate_pairwise_skew_stats(
                data, type="corrected"
            )
            plot(pairwise_error, ax[i])
            print("i is ", i)
            data = next(generator)

    except Exception as e:
        print(str(e)[:1000])
        print(type(e))
        pass

    fig.tight_layout(rect=[0.05, 0, 0.9, 0.95])
    fig.savefig("latency_hist_raw.png")
