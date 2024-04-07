import json
import logging
import argparse

import numpy as np

from .utils import (
    get_ntp_params_calltree,
    generate_call_tree,
    get_attribute_from_tags,
    get_attribute_idx_from_tags,
)


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


def correct_skew(tracedata, verbose=False):
    """Corrects the skew in a trace, only works on a pair of spans. Modifies in-place."""

    _preprocess_original_time(tracedata, verbose=verbose)

    # calibrate ntp params
    master_ntp_params = dict()
    for trace in tracedata:
        traceID, spans, processes, warnings = (
            trace["traceID"],
            trace["spans"],
            trace["processes"],
            trace["warnings"],
        )

        call_tree = generate_call_tree(trace)
        ntp_params = get_ntp_params_calltree(call_tree, spans)

        for parent, params in ntp_params.items():
            for theta, delta, child in params:
                if parent not in master_ntp_params:
                    master_ntp_params[parent] = dict()
                if child not in master_ntp_params[parent]:
                    master_ntp_params[parent][child] = []
                if child not in master_ntp_params:
                    master_ntp_params[child] = dict()
                if parent not in master_ntp_params[child]:
                    master_ntp_params[child][parent] = []
                master_ntp_params[parent][child].append((theta, delta))
                master_ntp_params[child][parent].append((-theta, delta))

    # Apply NTP Param filtering
    median_ntp_params = dict()
    for service, child_services in master_ntp_params.items():
        median_ntp_params[service] = {
            child_service: np.median(params, axis=0)
            for child_service, params in child_services.items()
        }

    # Apply clock skew correction
    for trace in tracedata:
        spans = trace["spans"]
        span_lookup = {span["spanID"]: span for span in spans}
        for span in spans:
            service = span["operationName"]
            for child in span["childSpanIds"]:
                child_span = span_lookup[child]
                theta, delta = median_ntp_params[service][child_span["operationName"]]
                span["startTime"] += theta / 2.0

                idx = get_attribute_idx_from_tags(span, "clock_skew_correction")
                if idx == -1:
                    span["tags"].append(
                        {"key": "clock_skew_correction", "value": theta / 2.0}
                    )
                else:
                    span["tags"][idx]["value"] += theta / 2.0

                child_span["startTime"] -= theta / 2.0
                idx = get_attribute_idx_from_tags(child_span, "clock_skew_correction")
                if idx == -1:
                    child_span["tags"].append(
                        {"key": "clock_skew_correction", "value": -theta / 2.0}
                    )
                else:
                    child_span["tags"][idx]["value"] -= theta / 2.0

    # Cast clock skew correction to string
    for trace in tracedata:
        spans = trace["spans"]
        for span in spans:
            idx = get_attribute_idx_from_tags(child_span, "clock_skew_correction")
            if idx != -1:
                child_span["tags"][idx]["value"] = str(
                    int(child_span["tags"][idx]["value"])
                )

    return tracedata
