import csv
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT.parent / "telecom-rag"
sys.path.insert(0, str(P1 / "src"))

from ingest import build_sections
from chunk import add_breadcrumbs, build_chunks

load_dotenv(ROOT / ".env")
client = OpenAI()

MODEL = "gpt-4o-mini"
PDF = P1 / "data" / "raw" / "ts_138300v190200p.pdf"
EVAL = ROOT / "eval" / "hard.csv"
OUT = ROOT / "data" / "qa_raw.jsonl"

PROMPT = (
    "You write training data for a 3GPP 5G NR question-answering model. "
    "You are given one excerpt from the specification. Write exactly 2 question-answer pairs "
    "that the excerpt fully answers. "
    "Pair 1: a precise, formal question. "
    "Pair 2: a short, casual question, the way a busy engineer would type it into a search box, "
    "using abbreviations where natural. "
    "Each answer must use ONLY facts stated in the excerpt, be 1 to 4 sentences, "
    "use abbreviations exactly as written, and never expand an abbreviation unless the excerpt does. "
    "Do not mention 'the excerpt' or 'the text' in the question or the answer. "
    'Return JSON only: {"pairs": [{"question": "...", "answer": "..."}, '
    '{"question": "...", "answer": "..."}]}'
)

REF_LINE = re.compile(r"\[\d+\]\s+(?:3GPP|IETF|ITU|IEEE|ETSI)")

SKIP_CLAUSES = {"2", "3.1"}


def is_junk(chunk):
    if chunk["clause"] in SKIP_CLAUSES:
        return True
    return len(REF_LINE.findall(chunk["text"])) >= 3


def held_out_clauses():
    with open(EVAL, encoding="utf-8") as f:
        return {row["expected_clause"] for row in csv.DictReader(f)}


def already_done():
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f}


def generate(chunk):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": chunk["text"]},
        ],
    )
    pairs = json.loads(resp.choices[0].message.content)["pairs"]
    return [
        {"question": p["question"].strip(), "answer": p["answer"].strip()}
        for p in pairs
        if p.get("question") and p.get("answer")
    ]


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    sections = add_breadcrumbs(build_sections(str(PDF)))
    chunks = build_chunks(sections)

    blocked = held_out_clauses()
    train_chunks = [c for c in chunks if c["clause"] not in blocked and not is_junk(c)]
    print("total chunks:", len(chunks), "| held out:", len(chunks) - len(train_chunks),
          "| usable for training:", len(train_chunks))

    done = already_done()
    todo = [c for c in train_chunks if c["id"] not in done]
    if limit:
        todo = todo[:limit]

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            try:
                pairs = generate(c)
            except Exception as e:
                print("FAILED", c["id"], e)
                time.sleep(2)
                continue
            f.write(json.dumps({
                "chunk_id": c["id"],
                "clause": c["clause"],
                "breadcrumb": c["breadcrumb"],
                "text": c["text"],
                "pairs": pairs,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(i, "/", len(todo), c["id"], len(pairs), "pairs")