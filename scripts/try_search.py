"""Try the policy search by hand: python scripts/try_search.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops.rag import DocIndex  # noqa: E402

idx = DocIndex.from_dir()
print(f"{len(idx.chunks)} chunks indexed")

questions = [
    "what is the target for hours per flight",
    "how many trucks do we need per day",
    "what happens if a driver fails two audits",
    "why did drivers get a 2 dollar raise",
]
for q in questions:
    print(f"\nQ: {q}")
    for r in idx.search(q, k=3):
        print("   ->", r["source"])

print("\n--- Full text of the best match for 'weekly bonus for safe drivers' ---")
print(idx.search("weekly bonus for safe drivers", k=1)[0]["text"])