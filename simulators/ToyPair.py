"""A script for generating a few thousand traces for a 3-chain toy model."""

from time import time_ns, sleep

from pyjaegersim import Application, Microservice

skews = [-10, 8]  # clock skew in ms
nodes = {
    key: Microservice(key, int(skew * 1e6)) for key, skew in zip(["A", "B"], skews)
}
edges = {"A": [("B", 15)], "B": []}
# transform ms to ns
edges = {
    key: [(service, int(latency * 1e6)) for service, latency in edges[key]]
    for key in edges
}

for key in edges:
    edges[key] = [(nodes[service], latency) for service, latency in edges[key]]

toy_model = Application(
    name="Toy-2-Pair",
    frontends=[nodes["A"]],
    nodes=nodes,
    edges=edges,
    frontend_weights=[1.0],
)

start_time = int(time_ns())
for _ in range(1500):
    start_time += int(10 * 1e6)
    sleep(0.0000000000000001)  # Prevents jaeger queue saturation
    toy_model.generate_trace(verbose=False)
