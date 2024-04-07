import json
import logging
import argparse
import itertools

import numpy as np
import networkx as nx

from .utils import (
    get_ntp_params_pair,
    generate_call_tree,
    get_attribute_from_tags,
    get_attribute_idx_from_tags,
    discover_services,
)


# DELETE
def apply_ntp_symmetric(parentSpan, childSpan, theta, delta, only_parent=True):
    half_delta = int(0.5 * delta)
    half_theta = int(0.5 * theta)
    parent_start_time = childSpan["startTime"] + half_theta - half_delta
    parent_end_time = parentSpan["startTime"] + (parentSpan["duration"] * 1e3)
    child_end_time = childSpan["startTime"] + (childSpan["duration"] * 1e3)

    parent_end_time_new = child_end_time + half_theta + half_delta
    parentSpan["duration"] = (parent_end_time_new - parent_start_time) / 1e3

    if not only_parent:
        child_start_time = parentSpan["startTime"] - half_theta + half_delta
        child_end_time_new = parent_end_time - half_theta - half_delta
        childSpan["duration"] = (child_end_time_new - child_start_time) / 1e3
        childSpan["startTime"] = child_start_time

    return parentSpan, childSpan


def _preprocess_original_time(tracedata, verbose=True):
    """Preprocesses the original time tags from Toy Data if applicable. Modifies in-place."""

    # Detect original_start_time tags
    contains_original_start_tag = False
    for trace in tracedata:
        for span in trace["spans"]:
            if get_attribute_from_tags(span, "original_start_time") is not None:
                contains_original_start_tag = True
                break
    if contains_original_start_tag:
        complete_coverage = True
        for trace in tracedata:
            for span in trace["spans"]:
                try:
                    span["startTime"] = int(
                        get_attribute_from_tags(span, "original_start_time")
                    )
                except:
                    complete_coverage = False
                    pass
        if verbose:
            logging.info(
                "Detected original_start_time tags, Removing Toy Data Jaeger Skew Correction"
            )
            if not complete_coverage:
                logging.warning(
                    "Some spans do not have original_start_time tags, Please check the data"
                )


def generate_service_graph(tracedata):
    services = discover_services(tracedata)

    G = nx.Graph()
    for service in services:
        G.add_node(service)

    for trace in tracedata:
        traceID, spans, processes, warnings = (
            trace["traceID"],
            trace["spans"],
            trace["processes"],
            trace["warnings"],
        )
        span_lookup = {span["spanID"]: span for span in spans}

        for span in spans:
            parent_service = span["operationName"]
            for child in span["childSpanIds"]:
                child_span = span_lookup[child]
                child_service = child_span["operationName"]
                G.add_edge(parent_service, child_service)

    return G


def correct_skew(tracedata, verbose=False, seed=2024, anchor=None):
    """Corrects the skew in a trace, only works on a pair of spans. Modifies in-place."""

    _preprocess_original_time(tracedata, verbose=verbose)

    # Generate service graph
    G = generate_service_graph(tracedata)

    # generate thetas
    def get_thetas(G, tracedata):
        thetas = {i: {} for i in G.nodes}

        for trace in tracedata:
            call_tree = generate_call_tree(trace)
            span_lookup = {span["spanID"]: span for span in trace["spans"]}
            seen_queue = set()
            queue = [(k, call_tree[k]) for k in call_tree.keys()]
            while queue:
                parent, curr_tree = queue.pop()
                parent_span = span_lookup[parent]
                seen_children = set()

                children = [(i, curr_tree[i]) for i in curr_tree.keys()]
                while children:
                    child, _child_tree = children.pop(0)

                    if child in seen_children:
                        continue
                    seen_children.add(child)
                    child_span = span_lookup[child]

                    # Get theta
                    theta, _ = get_ntp_params_pair(parent_span, child_span)

                    ckey = child_span["operationName"]
                    pkey = parent_span["operationName"]
                    if ckey not in thetas[pkey]:
                        thetas[pkey][ckey] = [theta]
                    else:
                        thetas[pkey][ckey].append(theta)
                    if pkey not in thetas[ckey]:
                        thetas[ckey][pkey] = [-theta]
                    else:
                        thetas[ckey][pkey].append(-theta)

                    # add subchildren
                    for i in child_span["childSpanIds"]:
                        children.append((i, _child_tree[i]))

                    if child_span["operationName"] not in seen_queue:
                        if child in curr_tree:
                            queue.append((child, curr_tree[child]))
                            seen_queue.add(span_lookup[child]["operationName"])

        for parent in thetas:
            for child in thetas[parent]:
                thetas[parent][child] = int(np.mean(thetas[parent][child]))

        # Get "VIRTUAL" thetas.
        # These are thetas that are never observed as spans.
        # They are the thetas that are not in the tracedata.
        for clique in nx.connected_components(G):
            for i, j in itertools.combinations(clique, 2):
                if j not in thetas[i]:
                    print(f"Adding virtual theta between {i} and {j}")
                    path = nx.shortest_path(G, i, j)
                    theta = 0
                    for k in range(len(path) - 1):
                        theta += thetas[path[k]][path[k + 1]]
                    thetas[i][j] = theta
                    thetas[j][i] = -theta

        # _t = {i: list(thetas[i].keys()) for i in thetas}
        # print(f"Thetas: {_t}")
        return thetas

    def apply_correction(tracedata, correction_lookup):
        for trace in tracedata:
            spans = trace["spans"]
            for child_span in spans:
                idx = get_attribute_idx_from_tags(child_span, "clock_skew_correction")
                name = child_span["operationName"]
                if name not in correction_lookup:
                    continue
                if idx == -1:
                    child_span["tags"].append(
                        {
                            "key": "clock_skew_correction",
                            "value": str(int(correction_lookup[name])),
                        }
                    )
                else:
                    child_span["tags"][idx]["value"] = str(
                        int(child_span["tags"][idx]["value"]) + correction_lookup[name]
                    )

                child_span["startTime"] = child_span["startTime"] + int(
                    correction_lookup[name]
                )

    if anchor is None:
        anchor = np.random.RandomState(seed).choice(list(nx.center(G)))
    seen = set([anchor])  # set
    while len(seen) < len(G.nodes):
        if verbose:
            print("Seen: ", seen)
        thetas = get_thetas(G, tracedata)

        # print("Thetas: ", {k: list(thetas[k].keys()) for k in thetas})
        # find all neighbors of current seen set
        neighbors = set()
        for node in seen:
            neighbors.update(G.neighbors(node))
        neighbors -= seen

        correction_thetas = {i: 0 for i in seen}

        to_add_seen = set()
        for node in neighbors:
            __thetas = {i: thetas[node][i] / 1.0e6 for i in seen if i in thetas[node]}
            _thetas = [np.mean(thetas[node][i]) for i in seen if i in thetas[node]]
            if verbose:
                print("Node: ", node, " Thetas: ", __thetas)
            theta = int(np.mean(_thetas))
            correction_thetas[node] = theta
            to_add_seen.add(node)

        seen = seen.union(to_add_seen)

        # apply correction
        if verbose:
            print("Correction thetas: ", {k: correction_thetas[k] / 1.0e6 for k in seen})
        apply_correction(tracedata, correction_thetas)

        yield tracedata

    # Cast clock skew correction to string
    for trace in tracedata:
        spans = trace["spans"]
        for child_span in spans:
            idx = get_attribute_idx_from_tags(child_span, "clock_skew_correction")
            if idx != -1:
                child_span["tags"][idx]["value"] = str(
                    int(child_span["tags"][idx]["value"])
                )

    # get average global correction:
    corrections = dict()
    seen_nodes = set()
    for trace in tracedata:
        spans = trace["spans"]
        for child_span in spans:
            operation_name = child_span["operationName"]
            if operation_name not in corrections:
                correction = get_attribute_from_tags(
                    child_span, "clock_skew_correction"
                )
                correction = int(correction) if correction is not None else 0
                corrections[operation_name] = correction
                seen_nodes.add(operation_name)

    # get average correction
    # avg_correction = int(np.mean(list(corrections.values())))
    # average_correction = {i: -avg_correction for i in seen_nodes}
    # apply_correction(tracedata, average_correction)

    return tracedata
