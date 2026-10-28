# Qwen3.8-Flash-Next on Abyss

日本語: [README.ja.md](README.ja.md)

Qwen3.8-Flash-Next (llama.cpp's `qwen4exp`) runs on Abyss's fast path (`--glm-pack`) from v1.3.74, with its prompt
path, verify window and MTP drafter. The older engine path ([docs/maya/ORCA.md](../maya/ORCA.md)) still works; the
fast path is several times faster on prompts and faster on answers.

Measured with orcarouter's
[Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) IQ3_XXS
(two shards, 85.2 GB) on an AI PRO R9700 (32 GB, PCIe Gen4 x4), 94 GB of RAM (`STRATA_GLM_RAM_GB=60`) and the iGPU
lane. NVIDIA: not built or measured on this path yet.

## The model

- **48 layers**: 36 gated delta-net layers (its output gate a sigmoid) and, every fourth layer, 12 attention layers
  (24 query heads of 256 over 2 KV heads, q / k norms, RoPE on 64 dims, a sigmoid output gate).
- **4 residual streams** of 2560 mixed by low-rank hyper-connections (no norm tensors: the mixer is the norm).
- **Layer 1's per-layer embedding (PLE)**: n-gram hashes of the last tokens pick 16 rows of a 28.8 GB table, which
  stays on the SSD (read directly, about 1 ms a token).
- **QSA** on the attention layers: past 2,051 cells each query attends to 512 blocks of 4 cells picked by an indexer
  (up to 2,051 cells it is dense attention).
- **512 experts a layer, 10 routed**, plus a shared expert behind a sigmoid gate, all 640 wide.
- The GGUF has no MTP (draft) block; the drafter is a separate file (below).

## Setup

1. **The pack** (the dense weights as the GGUF stores them; nothing converted to BF16):

   ```sh
   .venv/bin/python tools/iq_pack.py --fast \
     --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
     --out /path/to/IQ3_XXS/pack-fast
   ```

   Both shards must be complete. The pack is ~300 MB; the experts and the PLE table are read from the GGUF.

2. **The drafter (optional, recommended)**: the MTP runtime files the older path uses, made the same way
   ([docs/maya/ORCA.md](../maya/ORCA.md#preparation): `tools/mtp_fetch.py`, `tools/mtp_pack.py --experts q2_0`,
   `tools/mtp_rt.py`, then `data/draft_vocab.bin` copied in). The directory holds `dense.bin`, `dense.txt`,
   `experts.bin` and `draft_vocab.bin` (~820 MB); the engine keeps them in VRAM (~1 GB with its cache).

3. **The config** (`abyss-<name>.json`, see [docs/SETTINGS.md](../SETTINGS.md)):

   ```json
   {
    "exe": "build-hip/strata",
    "args": ["--glm-pack", "/path/to/IQ3_XXS/pack-fast", "--max-context", "32768", "--prefill", "auto"],
    "tokenizer": "/path/to/IQ3_XXS/pack-fast/tokenizer",
    "model_name": "qwen3.8-flash-next-uncensored-iq3_xxs",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "port": 8082,
    "backend": "hip",
    "env": {
     "STRATA_GLM_RAM_GB": "60",
     "STRATA_Q4X_MTP": "/path/to/mtp/rt"
    }
   }
   ```

   Start it with `.venv/bin/python serve/server.py --engine strata --config abyss-<name>.json --port 8082` (the
   `--port` on the command line is the one used).

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `STRATA_Q4X_MTP=<dir>` | off | the drafter's runtime files; without it the decode drafts only from repeats (n-gram) |
| `STRATA_Q35_MTP_K` | 4 (this model) | drafts a step at most (1..5) |
| `STRATA_Q35_MTP_MIN` / `STRATA_DS4_DSPARK_THR` | 0.75 (this model) | a draft goes into the verify window only at or above this confidence |
| `STRATA_Q4X_MTP_FULLHEAD=1` | off | the drafter's head over every token instead of `draft_vocab.bin`'s 106K |
| `STRATA_Q4X_DENSE=1` | off | debug: dense attention everywhere instead of QSA's selection |
| `STRATA_Q4X_PLE_SYNC=1` | off | debug: the verify window reads its PLE rows before layer 0 instead of during it |
| `STRATA_GLM_NO_MTP=1` | off | the drafter off even with `STRATA_Q4X_MTP` set |

## Measured (R9700, 2026-10-28)

| | older path (v1.3.73) | fast path (v1.3.74) |
| --- | ---: | ---: |
| prompt, 30K tokens | 217 tok/s | 1,037 tok/s |
| answer, prose (Japanese / English) | 52 tok/s | 64 / 65 tok/s |
| answer, writing code | 66-85 tok/s | 86 tok/s |
| answer, copying code | | 100-107 tok/s |
| answer, no drafter | | 51-53 tok/s |
| a 31K-token prompt's passphrase | found | found |
| a conversation's next turn (10.8K tokens) | reused | reused: 10,802 tokens, 0.28 s |

Prose and code are greedy 300-token answers from the command line; the server's answers vary a little from run to
run (the iGPU lane and where the experts sit). The drafter's windows: 74% of the drafts accepted on prose, 86% on
code; a window costs 30 ms against 18.6 ms for a token alone.

Quality: against llama.cpp on 494 positions past 5,600 tokens, KL 0.026 / top-1 92.3% (the token path) and 0.035 /
90.9% (the prompt path); llama.cpp's own HIP-vs-CPU difference on this model is KL 0.045. Greedy answers with and
without drafts are the same tokens.

## Limits

- One GPU: a split across cards has not been tried with this model, and the drafter loads only when one card runs
  the whole model.
- `--uncensor` has no file for this model (the GGUF used here is already uncensored).
- NVIDIA builds of this path have not been checked.
