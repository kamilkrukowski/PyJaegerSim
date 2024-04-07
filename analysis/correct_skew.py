import argparse
import json

from pyskewadjuster import correct_skew

parser = argparse.ArgumentParser(
    description="JSON Trace Data to apply skew correction algorithm to"
)
parser.add_argument("file", type=str, action="store", help="Input file")

args = parser.parse_args()

data = None
with open(args.file, "r") as f:
    data = json.load(f)["data"]

data = correct_skew(data, verbose=True)
