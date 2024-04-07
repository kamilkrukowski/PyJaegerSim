"""A package for synthetic generation of jaeger trace data."""

__version__ = "1.0.0"

from .datastructs import Span, Trace
from .utils import save_traces
from .tracerprovider import TracerProvider
from .microservice import Microservice, Application
