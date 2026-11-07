telecom-lora-finetune

Question: for answering questions about 3GPP TS 38.300, is fine-tuning a small open model better than the retrieval-augmented generation (RAG) assistant in telecom-rag?

Short answer: not on faithfulness. Fine-tuning Qwen2.5-1.5B-Instruct with QLoRA changed how it answers (citations, brevity, speed) but did not make the answers more faithful than the same base model given retrieved excerpts. Retrieval is what moved the score.

All data here comes from a public 3GPP specification plus synthetic questions. No employer data, logs or KPIs are used.

Results

Same 39 hard questions and 5 out-of-scope questions as P1. Retrieval is P1's hybrid + rerank pipeline (gold clause is in the top-5 context for 39/39).

System	Faithful (of 39)	Over-refused	Off-topic answered	Avg time	Avg new tokens
1. Base alone (closed-book)	1	0	n/a	8.4 s	239
2. Base + P1 retrieval	30	0	0/5	3.6 s	96
3. Fine-tuned alone (closed-book)	1	0	n/a	1.5 s	42
4. Fine-tuned + P1 retrieval	30	0	0/5	1.7 s	40
Reference: P1 with gpt-4o-mini	35	0	0/5	not measured	not measured

Timings are from one Colab T4 session (fp16, greedy). The fine-tuned rows use the merged model.

What fine-tuning did change (open-book, 39 questions)
	Base + retrieval	Fine-tuned + retrieval
Cites excerpts as (S#)	0 / 39	39 / 39
Invented abbreviation expansions	1	0
Median answer length	60 words	23 words
Unfaithful answers	9	9 (4 shared)

An equal 30/39 does not mean equal quality. Shorter answers make fewer claims, so each claim has fewer chances to be flagged. Faithfulness here does not measure completeness.

See eval/failure.md for every failure with a hypothesised cause.

Method
Chunks come from P1's ingestion and chunking of TS 38.300 (955 chunks). Clauses 2 (references) and 3.1 (abbreviations) and reference-heavy chunks are skipped.
Hold-out: every clause used by the 39 eval questions is excluded from training (148 chunks held out, 807 usable), so no eval answer is in the training data.
Generation: gpt-4o-mini writes 2 question/answer pairs per chunk (790 chunks, 1,581 pairs).
Verification: gpt-4o checks each pair is supported by the chunk and the question stands alone. 1,397 pairs kept.
Preparation: near-duplicate questions removed, then split by clause (8% validation). Output: 2,612 training and 224 validation examples, a mix of three formats:
closed-book (question only)
open-book (gold chunk plus 4 distractors, answer ends with (S#))
refusal (off-topic question, "I could not find this in the provided specs.")
Training: QLoRA on Qwen2.5-1.5B-Instruct. 4-bit NF4, LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections, 2 epochs, lr 2e-4 cosine, effective batch 16, fp16, loss on answer tokens only. Colab free T4.
Evaluation: greedy decoding. Judge is P1's statement-by-statement faithfulness check (gpt-4o) plus a code check for invented abbreviation expansions.
Serving and memory benchmark

One Colab T4 session, same 44 inputs, greedy decoding. The 4-bit model is the merged fp16 model quantised to NF4 at load.

Model	Weights memory	Peak memory	Tokens/sec (open)	Avg time (open)	Faithful (of 39, open)
Base fp16	3.09 GB	3.29 GB	26.5	3.64 s	30 (earlier run)
Fine-tuned fp16 (merged)	3.10 GB	3.28 GB	23.9	1.68 s	30
Fine-tuned 4-bit	1.16 GB	1.35 GB	17.4	2.24 s	33
Merging the adapter changed nothing: the merged model fails the same 9 questions as the unmerged one.
4-bit cuts weight memory by about 63% and peak memory by about 59%.
4-bit generates about 27% fewer tokens per second than fp16 on this T4, because the weights are unpacked while generating. It saves memory, not time.
Fine-tuned vs base is faster only because answers are shorter (40 vs 96 tokens). Per-token speed is similar (23.9 vs 26.5 tokens/sec).
4-bit scored 33/39 against 30/39 for fp16. That is not evidence it is better: 7 questions flipped to faithful and 4 flipped to unfaithful, which is within the noise of one question being about 2.6 points. The safe conclusion is no measurable faithfulness loss from 4-bit.
Limitations
One specification (TS 38.300) and 39 synthetic questions. One question is about 2.6 points, so small differences are noise.
Closed-book cannot show knowledge injection. The eval clauses were held out of training on purpose, so systems 1 and 3 measure generalisation to unseen clauses only. Testing injection would need the held-out clauses in training (roughly 2 more GPU hours).
Latency comes from one Colab session on a T4. Earlier runs in other sessions showed different absolute times for the same model, so compare rows within this table only.
LLM judge (gpt-4o). The faithfulness check is strict and judges closed-book answers against the gold chunk only.
Shorthand leakage: the tuned model copies casual abbreviations ("MR", "cand cells") from training answers to casual questions. The judge's expansion check does not detect this.
Retrieval-side failures remain (sibling clauses with near-identical text) and fine-tuning does not fix them.
Training data was generated with gpt-4o-mini; check the provider's terms before redistributing the dataset or adapter.
Reproduce
conda create -n p3 python=3.11 && conda activate p3
pip install pymupdf tiktoken openai python-dotenv
python src/make_dataset.py
python src/verify_dataset.py
python src/prepare_data.py
python src/export_retrieval.py      # needs the telecom-rag index
# Colab (T4): python src/train.py --epochs 2 --out <adapter dir>
# Colab (T4): python src/generate.py --name ft --adapter <adapter dir> --inputs eval_inputs.json
# Colab (T4): python src/benchmark.py --adapter <adapter dir>   # base16 / ft16 / ft4bit
python src/judge.py ft
python src/judge.py ft16
python src/judge.py ft4bit