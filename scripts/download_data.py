"""Download the benchmark files used in the paper into data/ and verify their SHA-256.

    python scripts/download_data.py              # LoCoMo + LongMemEval_S
    python scripts/download_data.py --only locomo

The datasets are not redistributed in this repository; see each dataset's own license.
"""
from __future__ import annotations
import argparse, hashlib, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    # snap-research/locomo (Maharana et al., ACL 2024), 10 conversations / 1,986 QA
    "locomo": ("data/locomo10.json",
               "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
               "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"),
    # xiaowu0162/longmemeval (Wu et al., ICLR 2025), LongMemEval_S, pinned revision
    "lme": ("data/longmemeval_s.json",
            "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/2ec2a557f339b6c0369619b1ed5793734cc87533/longmemeval_s",
            "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"),
}

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(name: str) -> bool:
    rel, url, digest = FILES[name]
    path = os.path.join(ROOT, rel)
    if os.path.exists(path) and sha256(path) == digest:
        print(f"[ok] {rel} already present"); return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"[..] {rel} <- {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256(tmp)
    if got != digest:
        os.remove(tmp)
        print(f"[!!] {rel}: sha256 mismatch ({got} != {digest})"); return False
    os.replace(tmp, path)
    print(f"[ok] {rel}"); return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FILES), action="append")
    a = ap.parse_args()
    ok = all([fetch(n) for n in (a.only or FILES)])
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
