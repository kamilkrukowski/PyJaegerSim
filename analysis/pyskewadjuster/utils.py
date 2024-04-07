from typing import List


def generate_call_tree(trace):
    """Generates a span call tree from a trace."""
    spans = trace["spans"]
    spanids = [i["spanID"] for i in spans]

    # Collect all edge-relationships in the trace
    child_relationships = {}
    for span in spans:
        if span["spanID"] not in child_relationships:
            child_relationships[span["spanID"]] = set()
        curr_children = set(span["childSpanIds"])
        child_relationships[span["spanID"]] = curr_children.union(
            set(span["childSpanIds"])
        )
    for span in child_relationships.keys():
        child_relationships[span] = list(child_relationships[span])

    # Find roots of the call tree
    call_tree = {}
    all_children = set()
    for some_children in child_relationships.values():
        all_children.update(some_children)
    for span in spanids:
        if span not in all_children:
            call_tree[span] = {}

    # Breadth first search to build the call tree
    queue = [(i, call_tree[i]) for i in call_tree.keys()]
    while queue:
        span, curr_tree = queue.pop(0)
        for child in child_relationships[span]:
            curr_tree[child] = {}
            queue.append((child, curr_tree[child]))

    return call_tree


def discover_services(traces: List):
    """Discovers all services in a list of traces."""
    services = set()
    for trace in traces:
        for span in trace["spans"]:
            services.add(span["operationName"])
    return services


def get_ntp_params_pair(parentSpan, childSpan):
    """Calculates the NTP parameters for a pair of spans."""
    t0, t3 = parentSpan["startTime"], parentSpan["startTime"] + (
        parentSpan["duration"] * 1e3
    )
    t1, t2 = childSpan["startTime"], childSpan["startTime"] + (
        childSpan["duration"] * 1e3
    )

    theta = 0.5 * ((t1 - t0) - (t2 - t3))
    delta = (t3 - t0) - (t2 - t1)
    return theta, delta


def get_ntp_params_calltree(call_tree, spans):
    """Calculates the NTP parameters for a trace."""
    ntp_params = {}
    parent_queue = list(call_tree.keys())
    span_lookup = {span["spanID"]: span for span in spans}
    while parent_queue:
        parent = parent_queue.pop(0)
        parentSpan = span_lookup[parent]
        for child in parentSpan["childSpanIds"]:
            childSpan = span_lookup[child]
            parent_queue.append(child)
            if parentSpan["operationName"] not in ntp_params:
                ntp_params[parentSpan["operationName"]] = []
            theta, delta = get_ntp_params_pair(parentSpan, childSpan)
            ntp_params[parentSpan["operationName"]].append(
                (theta, delta, childSpan["operationName"])
            )
    return ntp_params


def get_attribute_from_tags(span, attribute, default=None):
    """Gets an attribute from the tags of a span."""
    for tag in span["tags"]:
        if tag["key"] == attribute:
            return tag["value"]
    return default


def get_attribute_idx_from_tags(span, attribute):
    """Gets the index of an attribute from the tags of a span."""
    for idx, tag in enumerate(span["tags"]):
        if tag["key"] == attribute:
            return idx
    return -1
