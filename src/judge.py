import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
client = OpenAI()

JUDGE_MODEL = "gpt-4o"
REFUSAL = "could not find this in the provided specs"

JUDGE_PROMPT = (
    "You are checking an answer written from numbered excerpts of a 3GPP specification. "
    "First list every factual statement in the answer, including any expansion of an "
    "abbreviation (for example 'ABC (Alpha Beta Gamma)'). For each one write SUPPORTED if "
    "the excerpts state it or clearly imply it, otherwise UNSUPPORTED. "
    "Do not judge introductory or framing sentences (for example 'The steps are as "
    "follows:') or sentences about what the excerpts lack or what is missing; they are "
    "not factual statements and must not be listed. "
    "An abbreviation expansion is SUPPORTED only if it appears in the excerpts. "
    "End with a final line that is exactly 'VERDICT: YES' if no statement is UNSUPPORTED, "
    "otherwise 'VERDICT: NO'."
)

EXPANSION = re.compile(
    r"\b([A-Z][A-Za-z0-9\-]{1,10})\s*\(([A-Z][A-Za-z\-]+(?:\s+[A-Za-z\-]+){1,6})\)"
)


def is_refusal(text):
    return REFUSAL in text.lower()


def bad_expansions(text, excerpts):
    context = " ".join(excerpts).lower()
    return [f"{abbr} ({exp})" for abbr, exp in EXPANSION.findall(text) if exp.lower() not in context]


def faithful(answer, excerpts):
    context = "\n\n".join(f"(S{i + 1}) {t}" for i, t in enumerate(excerpts))
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nAnswer:\n{answer}"},
        ],
    )
    reply = resp.choices[0].message.content.strip()
    last_line = reply.splitlines()[-1].upper()
    bad = bad_expansions(answer, excerpts)
    if bad:
        return False, "UNSUPPORTED EXPANSIONS: " + "; ".join(bad)
    return "VERDICT: YES" in last_line, reply


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score(mode, inputs, answers):
    by_id = {item["id"]: item for item in inputs}
    refused = answered = faithful_count = 0
    oos_total = oos_false = 0
    seconds, tokens, details = [], [], []

    for a in answers:
        item = by_id[a["id"]]
        seconds.append(a["seconds"])
        tokens.append(a["new_tokens"])

        if item["kind"] == "out_of_scope":
            oos_total += 1
            if not is_refusal(a["answer"]):
                oos_false += 1
            continue

        if is_refusal(a["answer"]):
            refused += 1
            details.append({"id": a["id"], "verdict": "refused", "answer": a["answer"]})
            continue

        if mode == "closed":
            excerpts = [item["gold_text"]]
        else:
            excerpts = [c["text"] for c in item["contexts"]]
        ok, reason = faithful(a["answer"], excerpts)
        answered += 1
        faithful_count += ok
        details.append({
            "id": a["id"],
            "verdict": "faithful" if ok else "unfaithful",
            "question": a["question"],
            "answer": a["answer"],
            "reason": "" if ok else reason,
        })
        print(mode, a["id"], "faithful" if ok else "UNFAITHFUL")

    summary = {
        "mode": mode,
        "answerable": refused + answered,
        "over_refused": refused,
        "faithful": faithful_count,
        "answered": answered,
        "out_of_scope_false_answers": oos_false,
        "out_of_scope_total": oos_total,
        "mean_seconds": round(sum(seconds) / len(seconds), 2),
        "mean_new_tokens": round(sum(tokens) / len(tokens), 1),
    }
    return summary, details


if __name__ == "__main__":
    name = sys.argv[1]
    inputs = load(ROOT / "eval" / "eval_inputs.json")

    report = {}
    for mode in ("closed", "open"):
        answers = load(ROOT / "eval" / f"answers_{name}_{mode}.json")
        summary, details = score(mode, inputs, answers)
        report[mode] = {"summary": summary, "details": details}
        print("\n" + json.dumps(summary, indent=1) + "\n")

    with open(ROOT / "eval" / f"scores_{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("saved eval/scores_" + name + ".json")