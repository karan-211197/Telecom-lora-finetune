import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "qa_verified.jsonl"
TRAIN_OUT = ROOT / "data" / "train.jsonl"
VAL_OUT = ROOT / "data" / "val.jsonl"

SEED = 42
VAL_FRACTION = 0.08
N_DISTRACTORS = 4
REFUSAL = "I could not find this in the provided specs."

CLOSED_SYSTEM = "You answer questions about 3GPP 5G NR specifications."
OPEN_SYSTEM = (
    "You answer questions about 3GPP 5G NR specifications. "
    "Use ONLY the numbered context excerpts provided. "
    "Cite the excerpts you used like (S1) or (S2)(S3). "
    "Use abbreviations exactly as written; never expand or define an abbreviation "
    "unless an excerpt does so. "
    "If the excerpts contain relevant but partial information, answer with what they say "
    "and briefly state what is missing, using only terms from the excerpts. "
    "Only if they contain nothing relevant, reply exactly: "
    "'I could not find this in the provided specs.' "
    "Do not use outside knowledge."
)

OFF_TOPIC = [
    "Who invented the telephone?",
    "What is the tallest mountain in the world?",
    "How do I bake sourdough bread?",
    "What is the boiling point of water at sea level?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the best laptop for video editing?",
    "How many players are on a football team?",
    "What is the population of Canada?",
    "How do I reset my Wi-Fi router password?",
    "What is the price of a prepaid SIM card in India?",
    "Which smartphone has the best camera?",
    "How do I install Python on Windows?",
    "What is the capital of Australia?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "How do I train a puppy to sit?",
    "What is the exchange rate between dollars and euros?",
    "Which cloud provider is the cheapest?",
    "How do I write a resume for a software job?",
    "What year did the first iPhone launch?",
    "How does a refrigerator work?",
    "What is the longest river in Africa?",
    "How do I configure a Juniper switch?",
    "Who is the current CEO of Ericsson?",
    "What is the best way to learn guitar?",
]


def question_key(q):
    q = q.lower().replace("\u2019", "'").replace("what's", "what is")
    words = re.sub(r"[^a-z0-9\- ]", "", q).split()
    return " ".join(w for w in words if w not in {"a", "an", "the"})


def dedupe(rows):
    best = {}
    for r in rows:
        k = question_key(r["question"])
        if k not in best or len(r["answer"]) > len(best[k]["answer"]):
            best[k] = r
    return list(best.values())


def split_by_clause(rows):
    per_clause = Counter(r["clause"] for r in rows)
    clauses = sorted(per_clause)
    random.Random(SEED).shuffle(clauses)
    target = int(len(rows) * VAL_FRACTION)
    val_clauses, count = set(), 0
    for c in clauses:
        if count >= target:
            break
        val_clauses.add(c)
        count += per_clause[c]
    train = [r for r in rows if r["clause"] not in val_clauses]
    val = [r for r in rows if r["clause"] in val_clauses]
    return train, val


def unique_chunks(rows):
    seen = {}
    for r in rows:
        seen.setdefault(r["chunk_id"], r)
    return list(seen.values())


def build_context(excerpts):
    return "\n\n".join(f"(S{i + 1}) {e['text']}" for i, e in enumerate(excerpts))


def closed_example(r):
    return {"messages": [
        {"role": "system", "content": CLOSED_SYSTEM},
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["answer"]},
    ]}


def open_example(r, pool, rng):
    others = [p for p in pool if p["clause"] != r["clause"]]
    excerpts = rng.sample(others, N_DISTRACTORS) + [r]
    rng.shuffle(excerpts)
    gold = excerpts.index(r) + 1
    return {"messages": [
        {"role": "system", "content": OPEN_SYSTEM},
        {"role": "user", "content": f"Context:\n{build_context(excerpts)}\n\nQuestion: {r['question']}"},
        {"role": "assistant", "content": f"{r['answer']} (S{gold})"},
    ]}


def refusal_example(question, pool, rng):
    excerpts = rng.sample(pool, N_DISTRACTORS + 1)
    return {"messages": [
        {"role": "system", "content": OPEN_SYSTEM},
        {"role": "user", "content": f"Context:\n{build_context(excerpts)}\n\nQuestion: {question}"},
        {"role": "assistant", "content": REFUSAL},
    ]}


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    with open(IN, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print("verified pairs:", len(rows))

    rows = dedupe(rows)
    print("after dedupe:", len(rows))

    train, val = split_by_clause(rows)
    print("train pairs:", len(train), "| val pairs:", len(val))

    rng = random.Random(SEED)
    train_pool, val_pool = unique_chunks(train), unique_chunks(val)

    train_ex = [closed_example(r) for r in train] + [open_example(r, train_pool, rng) for r in train]
    for q in OFF_TOPIC:
        for _ in range(2):
            train_ex.append(refusal_example(q, train_pool, rng))
    rng.shuffle(train_ex)

    val_ex = [closed_example(r) for r in val] + [open_example(r, val_pool, rng) for r in val]
    rng.shuffle(val_ex)

    write_jsonl(TRAIN_OUT, train_ex)
    write_jsonl(VAL_OUT, val_ex)
    print("train examples:", len(train_ex), "| val examples:", len(val_ex))