# Controllable Text Generation with Neurally-Decomposed Oracle

This repository contains the source code to reproduce the experiments in NeurIPS 2022 paper
[Controllable Text Generation with Neurally-Decomposed Oracle](https://arxiv.org/abs/2205.14219) by [Tao Meng](https://mtsomethree.github.io/), [Sidi Lu](http://sidilu.cn/), [Nanyun Peng](https://vnpeng.net/) and [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/).

We are now working on the camera ready and the codebase is not a in a stable version. If you come up with some technical issue, please feel free to leave an issue or send an email to the first author.

- ### Abstract

We propose a general and efficient framework to control auto-regressive generation models with NeurAlly-Decomposed Oracle (NADO). Given a pre-trained base language model and a sequence-level boolean oracle function, we propose to decompose the oracle function into token-level guidance to steer the base model in text generation. Specifically, the token-level guidance is approximated by a neural model trained with examples sampled from the base model, demanding no additional auxiliary labeled data. We present the closed-form optimal solution to incorporate the token-level guidance into the base model for controllable generation. We further provide a theoretical analysis of how the approximation quality of NADO affects the controllable generation results. Experiments conducted on two applications: (1) text generation with lexical constraints and (2) machine translation with formality control demonstrate that our framework efficiently guides the base model towards the given oracle while maintaining high generation quality.

- ### Experiments

This repository will contain both experiments described in this paper. So far the LCG part is still under construction and expected to come out later October.

- ### Data

The machine translation formality change experiments leverage the [CALLHOME Spanish-English Speech Translation Corpus](https://aclanthology.org/2013.iwslt-papers.14/) as source data, and evaluate the BLUE score with the [fluent references](https://aclanthology.org/N19-1285/). Note that LDC access is required for the first dataset.

- ### Running experiments

**Requirements**

On systems where the system Python is externally managed (PEP 668, e.g. Debian/Ubuntu), install the dependencies in a virtual environment:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Activate the environment (`source .venv/bin/activate`) in every new shell before running the experiments.

**Running**

```bash
python train_MT.py
```

The code will automatically download [MarianMT model](https://huggingface.co/docs/transformers/model_doc/marian) and sample translated texts from source texts from Fisher-and-Callhome Corpus. The sampled data will be dumped in ./dump/MT directory. The sampled data is labeled by an formality oracle trained in [FUDGE paper](https://arxiv.org/abs/2104.05218). A NADO model will be trained by those labeled sampled data. The translated results will be evaluated based on oracle scores and the BLEU scores compared to fluent references.

```bash
Alternative arguments:
  --sample_batch_size  the batch size in sampling. Must be integer times of 8.
  --batch_size  the batch size in training. Must be a divider of sample_batch_size
  --regularization  the strength of regularization
  --max_length  the maximum length accepted in training or evaluation
```

- ### Toxicity (RealToxicityPrompts)

NADO on FUDGE's toxicity task, so the two methods can be compared on the same prompts, oracle and metrics. Base models are decoder-only: GPT-2 medium and the newer families Llama 3.2, Gemma 4 and Qwen2.5 (the `models` entries of `config.yaml`).

**Setup.** Llama and Gemma are gated: first accept their license on huggingface and set `HF_TOKEN` (or run `hf auth login`). Then run `recas/setup.sh` from the repository root, or submit `recas/setup.sub` on ReCaS. Besides the machine translation setup, it:

- creates `.venv-new`, with transformers 5 (`requirements-new.txt`). The newer families need it for Gemma 4, but `train_MT.py` doesn't support it; GPT-2 runs in `.venv`.
- downloads RealToxicityPrompts into `data/RealToxicityPrompts/prompts.jsonl`.
- downloads every base model of `config.yaml`, plus GPT-2 large for perplexity, into the huggingface cache under `.cache/huggingface` (`recas/toxicity.sh` reads them from there). Models it can't download, e.g. gated ones without a token, are skipped with a warning.
- links the fudge repo, if it sits next to this one (`ln -s ../fudge fudge`), and warns about missing oracles.

The oracles are the FUDGE toxicity predictors `fudge/ckpt/toxicity/<family>/model_best.pth.tar`, trained in the fudge repo. Each model uses the same per-family predictor FUDGE evaluates it with. A model whose predictor doesn't exist yet can't run (e.g. `llama32` until it's trained in fudge). Only fudge's files are read; no fudge code is run.

**Running.**

```bash
python train_toxicity.py --model gpt2_medium                    # in .venv
.venv-new/bin/python train_toxicity.py --model qwen25_0_5b      # newer llms
```

or `./run_toxicity.sh` to run every model (each in the venv of its config entry, see the header of the script for options, e.g. `MODELS`, `NUM_TEST_PROMPTS`, `EXTRA_ARGS`), or `recas/toxicity.sh` / `recas/toxicity.sub` on HTCondor.

`train_toxicity.py` follows `train_MT.py`:

1. **Sampling.** 8 continuations (`--samples_per_prompt`) of 20 tokens (`--length_cutoff`, as in FUDGE) are sampled from the base model for each of the first 10000 train prompts (`--num_train_prompts`). Each is labeled 1 if the oracle finds it non-toxic (`--target toxic` for the other way). Empty continuations count as not satisfying the oracle. Samples are cached in `dump/toxicity/<model>/`.
2. **Training.** R_c is trained on the samples. After each epoch, greedy NADO decoding of 500 validation prompts picks the best epoch by how many generations satisfy the oracle.
3. **Generation.** Greedy NADO decoding of the test prompts: one continuation per line, the same format as fudge's `evaluate_toxicity.py`. The script then computes the same metrics as fudge's `eval_toxicity_metrics.py`: average P(toxic) and fraction toxic according to the oracle, distinct-1/2/3, and perplexity under GPT-2 large. They are printed, and saved with the model statistics in `logs/model_stats/`.

The train/val/test split is recomputed from `prompts.jsonl` exactly as fudge's `prepare_toxicity_data.py` makes it, so the test prompts are fudge's and NADO never trains on them (if fudge's `test_prompts` file is there, `--test_prompts` checks it is the same split). `--constraint_factor 0` skips steps 1–2 and decodes with the base model alone (the greedy baseline, equal to FUDGE with `--condition_lambda 0`). `--load_dir` loads a trained R_c instead of training one.

**R_c for decoder-only models** (`causalLM.py`). The base model is frozen. R_c is a copy of its transformer plus a head giving, for every candidate next token, the logit of R(y_≤i). The head starts at 0, so R = 0.5 everywhere at first. Decoding uses q(y_i | y_<i) ∝ p(y_i | y_<i) · R(y_≤i)^λ (λ = `--constraint_factor`). The loss is the same as in `marianMT.ConstrainedIndMT`: BCE on the sampled sequences plus the KL regularizer (`--regularization`). R_c's embeddings are shared with the base model and frozen, and its trainable weights are kept in fp32. To fit the larger models on one GPU (~16GB), the config uses two options for them:

- `rc_layers`: R_c keeps only the first N decoder layers of the base model.
- `rc_head: tied`: instead of a new hidden × vocab linear layer (up to 670M parameters with Gemma's vocabulary), the head reuses the frozen LM head with a trainable scale and per-token bias.

GPT-2 uses a full copy and a new linear head, as in the paper. Peak GPU memory with the config's settings is ~9–14GB, except `gemma4_e4b`: its base model alone is ~16GB in bf16, so it needs a whole free 24GB GPU.

Note that the bf16 models give slightly different greedy outputs with batched decoding (`--test_batch_size`, prompts left-padded) than with one prompt at a time, as FUDGE decodes. Use `--test_batch_size 1` for outputs identical to FUDGE's (e.g. for the λ = 0 baseline).
