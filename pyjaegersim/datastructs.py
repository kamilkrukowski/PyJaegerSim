from typing import List
from time import time

import json
import os, binascii


class Span:
    """A span represents a single operation within a trace."""

    def __init__(self, serviceName=None, spanID=None, **kwargs):
        if spanID is None:
            spanID = binascii.hexlify(os.urandom(8)).decode("utf-8")
        self.spanID = spanID
        self.startTime = kwargs.get("startTime", int(time() * 1000000))
        self.duration = kwargs.get("duration", 10000)
        self.references = []
        self.serviceName = serviceName
        self.hasChildren = False
        self.childSpanIDs = []

        self.process = {
            "serviceName": serviceName,
            "tags": [
                {"key": "telemetry.sdk.language", "value": "python"},
                {"key": "telemetry.sdk.name", "value": "opentelemetry"},
                {"key": "telemetry.sdk.version", "value": "1.24.0"},
                {"key": "service.name", "value": serviceName},
            ],
        }
        self.processID = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = int(duration)

    def make_span(self, *args, **kwargs):
        """Child Span constructor."""
        kwargs["parentID"] = self.spanID
        self.hasChildren = True
        outSpan = Span(*args, **kwargs)
        self.childSpanIDs.append(outSpan.spanID)
        return outSpan

    def __str__(self):
        return json.dumps(self.__dict__)

    def to_dict(self):
        return self.__dict__


class Trace:
    """A collection of spans."""

    def __init__(self, traceID=None):
        if traceID is None:
            traceID = binascii.hexlify(os.urandom(16)).decode("utf-8")
        self.traceID = traceID
        self.spans = []
        self.processes = {}

    def add_span(self, span: Span):
        if not hasattr(self, "spans"):
            self.spans = []
        span.traceID = self.traceID
        self.spans.append(span.to_dict())
        process_id = len(self.processes) + 1
        span.processID = f"p{process_id}"
        self.processes[span.processID] = {
            "serviceName": span.serviceName,
        }

    def add_spans(self, spans: List[Span]):
        for span in spans:
            self.add_span(span)

    def __str__(self):
        return json.dumps(self.__dict__)
