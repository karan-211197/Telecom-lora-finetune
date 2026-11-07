import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
client = OpenAI()

JUDGE_MODEL = "gpt-4o"
IN = ROOT / "data" / "qa_raw.jsonl"
KEPT = ROOT / "data" / "qa_verified.jsonl"
REJECTED = ROOT / "data" / "qa_rejected.jsonl"

JUDGE_PROMPT = (
    "You check training pairs for a 3GPP 5G NR question-answering model. "
    "You get one specification excerpt and a numbered list of question-answer pairs. "
    "For each pair decide two things. "
    "SUPPORTED: every factual statement in the answer, including any expansion of an "
    "abbreviation, is stated in the excerpt or clearly implied by it. "
    "Answers that use outside knowledge, or mix up neighbouring entries, are not supported. "
    "STANDALONE: the question makes sense without seeing the excerpt, and the answer is "
    "meaningful on its own (no 'also', 'this', 'it' that points back to unseen text). "
    'Return JSON only: {"results": [{"supported": true, "standalone": true, "reason": "short"}, ...]} '
    "with exactly one result per pair, in the same order."
)


def judge(row):
    numbered = "\n".join(
        f"{i + 1}. Q: {p['question']}\n   A: {p['answer']}" for i, p in enumerate(row["pairs"])
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Excerpt:\n{row['text']}\n\nPairs:\n{numbered}"},
        ],
    )
    results = json.loads(resp.choices[0].message.content)["results"]
    if len(results) != len(row["pairs"]):
        raise ValueError("judge returned a different number of results than pairs")
    return results


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    with open(IN, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]

    kept = rejected = failed = 0
    with open(KEPT, "w", encoding="utf-8") as fk, open(REJECTED, "w", encoding="utf-8") as fr:
        for i, row in enumerate(rows, 1):
            try:
                results = judge(row)
            except Exception as e:
                failed += 1
                print("FAILED", row["chunk_id"], e)
                time.sleep(2)
                continue
            for pair, res in zip(row["pairs"], results):
                record = {
                    "chunk_id": row["chunk_id"],
                    "clause": row["clause"],
                    "breadcrumb": row["breadcrumb"],
                    "text": row["text"],
                    "question": pair["question"],
                    "answer": pair["answer"],
                }
                if res.get("supported") and res.get("standalone"):
                    fk.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    record["reason"] = res.get("reason", "")
                    fr.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rejected += 1
            if i % 50 == 0:
                print(i, "/", len(rows), "| kept", kept, "| rejected", rejected)

    print("done | kept", kept, "| rejected", rejected, "| chunks failed", failed)