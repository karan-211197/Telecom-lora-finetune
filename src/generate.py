import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import CLOSED_SYSTEM, OPEN_SYSTEM, build_context


def make_messages(item, mode):
    if mode == "closed":
        return [
            {"role": "system", "content": CLOSED_SYSTEM},
            {"role": "user", "content": item["question"]},
        ]
    context = build_context(item["contexts"])
    return [
        {"role": "system", "content": OPEN_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {item['question']}"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--inputs", default="eval_inputs.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    with open(args.inputs, encoding="utf-8") as f:
        items = json.load(f)

    for mode in ("closed", "open"):
        results = []
        for item in items:
            inputs = tokenizer.apply_chat_template(
                make_messages(item, mode),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            start = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            seconds = time.time() - start
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            results.append({
                "id": item["id"],
                "kind": item["kind"],
                "question": item["question"],
                "answer": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                "seconds": round(seconds, 2),
                "new_tokens": int(len(new_tokens)),
            })
            print(mode, item["id"], f"{seconds:.1f}s")
        path = f"answers_{args.name}_{mode}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print("saved", path)


if __name__ == "__main__":
    main()