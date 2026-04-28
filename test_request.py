"""
Smoke-test script for the running Flask API.

Usage:
    # With the server running on localhost:8000:
    python test_request.py

    # Or point at a different host:
    API_URL=http://somehost:8000 python test_request.py
"""
import json
import os
import sys
import urllib.request


API_URL = os.getenv("API_URL", "http://localhost:8000")


SAMPLE_PAYLOAD = {
    "question": (
        "Write pseudocode for a function that takes a comma-separated string of "
        "integers, parses it into a list, and returns the sum of all even numbers."
    ),
    "answer": (
        "function sumEvens(s):\n"
        "    parts = split(s, ',')\n"
        "    total = 0\n"
        "    for p in parts:\n"
        "        n = toInt(p)\n"
        "        if n mod 2 == 0:\n"
        "            total = total + n\n"
        "    return total"
    ),
    "criteria": [
        {
            "name": "Input parsing",
            "max_score": 2,
            "description": (
                "Criterion 1 - Input parsing (commas) (2 marks). "
                "2 marks: splits input correctly and converts each piece to an integer. "
                "1 mark: splits but does not convert, OR converts without splitting. "
                "0 marks: no attempt at parsing."
            ),
        },
        {
            "name": "Even detection",
            "max_score": 2,
            "description": (
                "Criterion 2 - Even detection (2 marks). "
                "2 marks: uses modulo (or equivalent) to correctly detect even numbers. "
                "1 mark: detection logic present but flawed. "
                "0 marks: no even-detection logic."
            ),
        },
        {
            "name": "Accumulation and return",
            "max_score": 1,
            "description": (
                "Criterion 3 - Accumulation & return (1 mark). "
                "1 mark: running total is accumulated and returned. "
                "0 marks: missing accumulation or missing return."
            ),
        },
    ],
}


def call(url, method="GET", body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main():
    print(f"→ GET {API_URL}/health")
    status, body = call(f"{API_URL}/health")
    print(f"  [{status}] {body}\n")

    print(f"→ GET {API_URL}/info")
    status, body = call(f"{API_URL}/info")
    print(f"  [{status}] {json.dumps(body, indent=2)}\n")

    print(f"→ POST {API_URL}/predict")
    status, body = call(f"{API_URL}/predict", method="POST", body=SAMPLE_PAYLOAD)
    print(f"  [{status}]")
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Could not reach {API_URL}: {e}", file=sys.stderr)
        sys.exit(1)
