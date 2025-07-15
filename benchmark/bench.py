#!/bin/python

from safetensors import safe_open
import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark safetensors file")
    parser.add_argument("path", type=str, help="Path to the safetensors file")
    parser.add_argument("loop", type=int, default=1, help="Number of times to loop through the file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for loading tensors (default: cpu)")
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    with safe_open(args.path, framework="pt", device=args.device) as f:
        for i in range(args.loop):
            results = {}
            for key in f.keys():
                results[key] = f.get_tensor(key)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time) / args.loop
    print(f"Benchmark completed in {duration}")
