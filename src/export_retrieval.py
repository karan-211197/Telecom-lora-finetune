import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT.parent / "telecom-rag"
sys.path.insert(0, str(P1 / "src"))
os.chdir(P1)

from pipeline import retrieve_best
from rag import col

HARD = ROOT / "eval" / "hard.csv"
SYNTH = ROOT / "eval" / "synthetic.csv"
OUT = ROOT / "eval" / "eval_inputs.json"


def load(path, kind):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["type"] == kind]


def pack(hits):
    return [{"text": h["text"], "clause": h["clause"], "page": h["page"]} for h in hits]


if __name__ == "__main__":
    items = []

    for r in load(HARD, "synthetic"):
        gold = col.get(ids=[r["expected_chunk"]], include=["documents"])["documents"][0]
        hits = retrieve_best(r["question"], k=5)
        items.append({
            "id": r["id"],
            "kind": "answerable",
            "question": r["question"],
            "expected_clause": r["expected_clause"],
            "gold_text": gold,
            "gold_in_context": any(h["clause"] == r["expected_clause"] for h in hits),
            "contexts": pack(hits),
        })
        print("answerable", r["id"], "done")

    for i, r in enumerate(load(SYNTH, "out_of_scope"), start=1):
        hits = retrieve_best(r["question"], k=5)
        items.append({
            "id": f"oos{i}",
            "kind": "out_of_scope",
            "question": r["question"],
            "contexts": pack(hits),
        })
        print("out_of_scope", i, "done")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    answerable = [x for x in items if x["kind"] == "answerable"]
    print("saved", len(items), "items |",
          "gold in top-5 context:", sum(x["gold_in_context"] for x in answerable), "/", len(answerable))