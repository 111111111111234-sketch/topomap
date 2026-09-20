#!/usr/bin/env python3
"""Replace non-finite GOAT SPL values with zero, preserving binary backups."""

import argparse
import glob
import math
import os
import pickle
import shutil


def sanitize(value):
    changed = 0
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key], item_changed = sanitize(item)
            changed += item_changed
        return result, changed
    if isinstance(value, list):
        result = []
        for item in value:
            clean, item_changed = sanitize(item)
            result.append(clean)
            changed += item_changed
        return result, changed
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        return 0.0, 1
    return value, 0


def process_directory(directory):
    total = 0
    patterns = ("spl_by_snapshot*.pkl", "spl_by_distance*.pkl", "spl_by_task*.pkl")
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            clean, changed = sanitize(payload)
            if not changed:
                continue
            backup = path + ".pre_spl_sanitize.bak"
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
            with open(path, "wb") as handle:
                pickle.dump(clean, handle)
            print(f"{path}: replaced {changed} non-finite value(s); backup={backup}")
            total += changed
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    args = parser.parse_args()
    changed = sum(process_directory(directory) for directory in args.directories)
    print(f"total replacements: {changed}")


if __name__ == "__main__":
    main()
