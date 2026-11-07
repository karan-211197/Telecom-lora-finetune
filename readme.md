# telecom-lora-finetune

**Question:** for answering questions about 3GPP TS 38.300, is fine-tuning a small open model better than the retrieval-augmented generation (RAG) assistant in [telecom-rag](https://github.com/karan-211197/telecom-rag)?

**Short answer:** not on faithfulness. Fine-tuning Qwen2.5-1.5B-Instruct with QLoRA changed *how* it answers (citations, brevity, speed) but did not make the answers more faithful than the same base model given retrieved excerpts. Retrieval is what moved the score.

> All data here comes from a public 3GPP specification plus synthetic questions. No employer data, logs or KPIs are used.

## Results

Same 39 hard questions and 5 out-of-scope questions as P1. Retrieval is P1's hybrid + rerank pipeline (gold clause is in the top-5 context for 39/39).

| System | Faithful (of 39) | Over-refused | Off-topic answered | Avg time | Avg new tokens |
|---|---|---|---|---|---|
| 1. Base alone (closed-book) | 1 | 0 | n/a | 9.8 s | 239 |
| 2. Base + P1 retrieval | 30 | 0 | 0/5 | 4.5 s | 95 |
| 3. Fine-tuned alone (closed-book) | 1 | 0 | n/a | 3.5 s | 42 |
| 4. Fine-tuned + P1 retrieval | 30 | 0 | 0/5 | 3.6 s | 40 |
| Reference: P1 with gpt-4o-mini | 35 | 0 | 0/5 | not measured | not measured |

### What fine-tuning did change (open-book, 39 questions)

| | Base + retrieval | Fine-tuned + retrieval |
|---|---|---|
| Cites excerpts as `(S#)` | 0 / 39 | 39 / 39 |
| Invented abbreviation expansions | 1 | 0 |
| Median answer length | 60 words | 23 words |
| Unfaithful answers | 9 | 9 (4 shared) |

An equal 30/39 does not mean equal quality. Shorter answers make fewer claims, so each claim has fewer chances to be flagged. Faithfulness here does not measure completeness.

See [`eval/failure.md`](eval/failure.md) for every failure with a hypothesised cause.

## Method

1. **Chunks** come from P1's ingestion and chunking of TS 38.300 (955 chunks). Clauses 2 (references) and 3.1 (abbreviations) and reference-heavy chunks are skipped.
2. **Hold-out:** every clause used by the 39 eval questions is excluded from training (148 chunks held out, 807 usable), so no eval answer is in the training data.
3. **Generation:** gpt-4o-mini writes 2 question/answer pairs per chunk (790 chunks, 1,581 pairs).
4. **Verification:** gpt-4o checks each pair is supported by the chunk and the question stands alone. 1,397 pairs kept.
5. **Preparation:** near-duplicate questions removed, then split by clause (8% validation). Output: 2,612 training and 224 validation examples, a mix of three formats:
   - closed-book (question only)
   - open-book (gold chunk plus 4 distractors, answer ends with `(S#)`)
   - refusal (off-topic question, "I could not find this in the provided specs.")
6. **Training:** QLoRA on Qwen2.5-1.5B-Instruct. 4-bit NF4, LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections, 2 epochs, lr 2e-4 cosine, effective batch 16, fp16, loss on answer tokens only. Colab free T4.
7. **Evaluation:** greedy decoding. Judge is P1's statement-by-statement faithfulness check (gpt-4o) plus a code check for invented abbreviation expansions.

## Serving and memory benchmark

_To be filled after the next GPU session: base fp16 vs fine-tuned fp16 vs fine-tuned 4-bit, with GPU memory, latency, tokens/sec and judge score for the 4-bit answers._

| Model | GPU memory | Tokens/sec | Faithful (of 39) |
|---|---|---|---|
| Base fp16 | TBD | TBD | TBD |
| Fine-tuned fp16 (merged) | TBD | TBD | TBD |
| Fine-tuned 4-bit | TBD | TBD | TBD |

## Limitations

- **One specification** (TS 38.300) and **39 synthetic questions**. One question is about 2.6 points, so small differences are noise.
- **Closed-book cannot show knowledge injection.** The eval clauses were held out of training on purpose, so systems 1 and 3 measure generalisation to unseen clauses only. Testing injection would need the held-out clauses in training (roughly 2 more GPU hours).
- **Latency compared across different Colab sessions.** The gap between systems 2 and 4 is mostly output length (95 vs 40 tokens), not per-token speed.
- **LLM judge** (gpt-4o). The faithfulness check is strict and judges closed-book answers against the gold chunk only.
- **Shorthand leakage:** the tuned model copies casual abbreviations ("MR", "cand cells") from training answers to casual questions. The judge's expansion check does not detect this.
- **Retrieval-side failures remain** (sibling clauses with near-identical text) and fine-tuning does not fix them.
- Training data was generated with gpt-4o-mini; check the provider's terms before redistributing the dataset or adapter.

## Reproduce

```
conda create -n p3 python=3.11 && conda activate p3
pip install pymupdf tiktoken openai python-dotenv
python src/make_dataset.py
python src/verify_dataset.py
python src/prepare_data.py
python src/export_retrieval.py      # needs the telecom-rag index
# Colab (T4): python src/train.py --epochs 2 --out <adapter dir>
# Colab (T4): python src/generate.py --name ft --adapter <adapter dir> --inputs eval_inputs.json
python src/judge.py ft
```

Set `OPENAI_API_KEY` in `.env`. The adapter and any `.pem` keys are not committed.