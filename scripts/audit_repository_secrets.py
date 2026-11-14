#!/usr/bin/env python3
"""Scan tracked content/reachable history; report locations and rule IDs, never values.

This detects common credential formats, not every possible secret. Ignored local
files, credential stores, GitHub Actions secrets and inaccessible refs are not read.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "private-key": rb"-{5}BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-{5}",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "aws-access-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "credential-literal": rb'''(?i)["']?(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)["']?\s*[:=]\s*["']([A-Za-z0-9][A-Za-z0-9_./+=-]{19,})["']''',
}
PATTERNS = {k: re.compile(v) for k, v in RULES.items()}


def findings(data):
    return [dict(rule=name, line=data.count(b"\n", 0, m.start())+1)
            for name, pattern in PATTERNS.items() for m in pattern.finditer(data)]


def git(*args, input=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, input=input)


def audit(history=False):
    hits, count = [], 0
    if not history:
        # Index content matches what can be committed; never open ignored credentials.
        names = git("ls-files", "-z").split(b"\0")
        for raw in names:
            if not raw:
                continue
            path = raw.decode("utf-8", "replace")
            data = git("show", ":"+path)
            count += 1
            hits.extend(dict(path=path, **hit) for hit in findings(data))
    else:
        objects = git("rev-list", "--objects", "--all").splitlines()
        paths = {}
        for record in objects:
            oid, _, name = record.partition(b" ")
            paths[oid] = name.decode("utf-8", "replace")
        ids = list(paths)
        metadata = git("cat-file", "--batch-check", input=b"\n".join(ids)+b"\n").splitlines()
        blobs = [line.split()[0] for line in metadata if line.split()[1] == b"blob"]
        # Bounded batches avoid persistent pipe deadlocks. No blob content is printed.
        for start in range(0, len(blobs), 20):
            batch = blobs[start:start+20]
            output = git("cat-file", "--batch", input=b"\n".join(batch)+b"\n")
            pos = 0
            for oid in batch:
                end = output.index(b"\n", pos)
                size = int(output[pos:end].split()[2])
                data = output[end+1:end+1+size]
                pos = end+size+2
                count += 1
                hits.extend(dict(object=oid.decode(), path=paths[oid], **hit) for hit in findings(data))
    return dict(scope="reachable_history" if history else "tracked_index", blobs_scanned=count,
                findings=hits, detected_count=len(hits),
                limitation="Pattern scan only; no guarantee that all secret formats are detected.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    result = audit(args.history)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["detected_count"] else 0)
