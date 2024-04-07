"""A package for calculating and correcting clock skew in Jaeger traces."""

__version__ = "0.3.0"

from .graph_corrector import correct_skew, generate_call_tree
from .utils import (
    generate_call_tree,
    discover_services,
    get_ntp_params_pair,
    get_attribute_from_tags,
    get_attribute_idx_from_tags,
)
