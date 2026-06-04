# LLM Exploit Generation Agent

An automated **exploit-generation agent** for discovering prompt-injection
vulnerabilities in locally hosted Large Language Model (LLM) applications.

> **ACADEMIC USE ONLY.** This repository is provided **strictly for academic
> security-evaluation research** in controlled, authorized, sandboxed
> environments. It must not be used to test, attack, or compromise any system
> without explicit permission from the system owner.

> **Disclaimer — no liability.** The authors and contributors of this project
> **are not liable** for any direct or indirect damages, losses, legal
> consequences, or misuse arising from the download, modification, or use of
> this software. You are solely responsible for ensuring your use complies with
> applicable laws, institutional policies, and ethical guidelines. By using this
> software, you agree that you do so at your own risk and only for legitimate
> academic research on systems you are authorized to evaluate.

---

## Academic use only

This project exists to study **automated vulnerability discovery methodology**
against a **local, sandboxed** target application. It is **not** intended for:

- Testing production, third-party, or otherwise unauthorized systems
- Developing, distributing, or deploying weaponized prompts
- Any commercial, offensive, or non-research purpose

If you are unsure whether your use case is permitted, **do not run this software**
until you have obtained appropriate authorization and guidance from your institution
or legal counsel.

---

## Overview

The project measures how effectively an LLM-driven mutation agent can induce a
sandboxed target application to leak secret **canary tokens** that are embedded
in its system prompt. A leak (the full canary string appearing verbatim in the
model's output) is counted as a successful prompt injection.

The headline metric is **Attack Success Rate (ASR)** — the fraction of trials in
which a canary token is leaked. The agent's ASR is compared against static and
random baselines to demonstrate that systematic, LLM-driven mutation
outperforms naive approaches.

### Core contribution

The research contribution is the **exploit-generation agent itself** — an
automated pipeline that systematically discovers prompt-injection
vulnerabilities more effectively than static or random baselines. The agent, not
any individual exploit, is the deliverable.

### Scope and constraints

- **Single exploit class:** prompt injection only.
- **Local / sandboxed only:** no testing of production or third-party systems.
- **Open / self-hosted models:** any model served by an Ollama-compatible backend.
- **Deterministic evaluation:** a verbatim canary-substring oracle, not an LLM judge.
- **Academic use only:** no publication of weaponized prompts; no non-research use.

---

## How it works

### The 3-round mutation pipeline

For a set of `N` starting prompts (from `data/starting_prompts.jsonl`), the runner executes:

- **Round 0 — Baseline.** Each starting prompt is sent to the target as-is.
- **Round 1 — Fork by technique.** Each Round-0 prompt is mutated once per
  technique (`logic_traps`, `roleplay_social`, `encoding`), producing 3 variants.
- **Round 2 — Random stacked follow-up.** One randomly chosen technique is
  layered on top of each Round-1 result, preserving the inner structure.

Total trials per run: `N + 3N + 3N = 7N`.

Every trial (prompt, technique, mutation path, target response preview, and
oracle verdict) is logged to `results/runs/<run_id>/trials.jsonl`, with an
aggregated `summary.json` and a `config_snapshot.yaml`.

### Components

```
.
├── configs/             # YAML configuration (experiment, target app, policy)
├── data/                # Starting prompts, fixed baseline, benign tasks, canaries, docs
├── src/
│   ├── targetapp/       # Sandbox target: RAG app + LLM wrapper + interfaces
│   ├── agent/           # prompt_datasets, mutators, rounds runner (core)
│   ├── evaluation/      # Deterministic canary oracle, metrics, labeling
│   ├── experiments/     # Experiment + baseline runners (and a sweep stub)
│   └── analysis/        # Result analysis / plotting helpers
├── results/runs/        # Per-run outputs (git-ignored; structure kept via .gitkeep)
├── experiment_visualization.py   # Build summary tables + ASR plots from runs
└── test_*.py            # Test / smoke-test scripts
```

| Module | Responsibility |
| --- | --- |
| `src/targetapp/local_chat_model.py` | Thin HTTP wrapper around the Ollama-compatible backend |
| `src/targetapp/rag_app.py` | Sandboxed RAG target application that embeds canary tokens |
| `src/agent/prompt_datasets.py` | Load JSONL prompt corpora; build comparison-baseline trial lists |
| `src/agent/mutators.py` | `LLMMutator` techniques + `Mutator` orchestration (Round 1 / 2) |
| `src/agent/runner.py` | `RoundsRunner` — the 3-round pipeline |
| `src/evaluation/oracle.py` | Deterministic verbatim-canary success oracle |
| `src/evaluation/metrics.py` | Per-round / per-technique / per-combination ASR aggregation |
| `src/experiments/run_experiment.py` | End-to-end pipeline entry point |
| `src/experiments/run_baselines.py` | Static and random-mutation baselines |

---

## Requirements

- **Python 3.11+**
- An **Ollama-compatible** LLM backend ([Ollama](https://ollama.com)) with at
  least one model installed on that backend.

The project talks to the backend over plain HTTP, so the model can run on the
same machine or on a remote host (see [Configuration](#configuration)).

---

## Setup

1. **Install the backend and pull a model of your choice:**

   ```bash
   # Install Ollama from https://ollama.com, then:
   ollama serve
   ollama pull <your-model-name>    # use any model you have rights to run
   ```

2. **Configure the model name** — replace the placeholder in
   `configs/targetapp.yaml`:

   ```yaml
   llm:
     model: "ENTER YOUR SELECTED MODEL HERE"   # <-- put your exact model name here
   ```

   The name must match what your backend reports (e.g. from `ollama list`).
   You may also set `OLLAMA_MODEL` in `.env` (see `.env.example`).

3. **Create a virtual environment and install the package:**

   ```bash
   python -m venv venv
   # Windows:        venv\Scripts\activate
   # macOS / Linux:  source venv/bin/activate

   pip install -e .            # add ".[dev]" for test/lint tooling
   ```

4. **(Optional) configure environment variables:**

   ```bash
   cp .env.example .env        # then edit as needed
   ```

---

## Configuration

The project is model- and host-agnostic. Nothing is hard-coded to a specific
model or to `localhost`.

### Choosing the model and host

- **Model:** replace `ENTER YOUR SELECTED MODEL HERE` in `configs/targetapp.yaml`
  (`llm.model`), or set the `OLLAMA_MODEL` environment variable. Some scripts
  also accept a `--model` flag. The repository ships with a placeholder only —
  you must configure a real model name before running experiments.
- **Host:** set `llm.base_url` in `configs/targetapp.yaml`, or the
  `OLLAMA_BASE_URL` environment variable. When both are unset it defaults to
  `http://localhost:11434`.

To run against a remote backend, for example:

```bash
export OLLAMA_BASE_URL=http://my-gpu-box:11434   # Windows: $env:OLLAMA_BASE_URL=...
```

### Config files

| File | Purpose |
| --- | --- |
| `configs/targetapp.yaml` | Target app type, model + backend settings, system prompt, RAG / canary settings |
| `configs/experiment.yaml` | Starting-prompt source + count, enabled rounds, techniques, evaluation, seed |
| `configs/policy.yaml` | Ethical guidelines, constraints, and deterministic detection rules |

### Prompt data files

| File | Role |
| --- | --- |
| `data/starting_prompts.jsonl` | Starting prompts for the **3-round agent** (`baseline.file` in `experiment.yaml`) |
| `data/fixed_baseline_prompts.jsonl` | Fixed attacks for the **static** comparison baseline (`run_baselines.py --static`) |
| `data/benign_tasks.jsonl` | Non-attack tasks (optional) |
| `data/canaries.json` | Canary tokens embedded in the target system prompt |

### Canary tokens

`data/canaries.json` contains randomly generated `CANARY_SECRET_<NN>_<8hex>`
strings used purely as leak markers. They are **not** secrets or credentials —
they are random research artifacts. Rotate them between experiments by
regenerating with `uuid.uuid4().hex[:8]` to avoid overfitting.

---

## Usage

Run all commands from the project root.

### Run the full 3-round pipeline

```bash
python src/experiments/run_experiment.py --config configs/experiment.yaml

# Fast smoke run on just a couple of base prompts:
python src/experiments/run_experiment.py --max-base-prompts 2
```

Outputs land in `results/runs/<run_id>/`.

### Run the baselines

```bash
# Both static and random-mutation baselines:
python src/experiments/run_baselines.py --out-dir results/runs

# Just one of them:
python src/experiments/run_baselines.py --static
python src/experiments/run_baselines.py --random --random-trials 50 --seed 42
```

### Visualize results

```bash
python experiment_visualization.py \
    --runs-root results/runs \
    --baseline-json results/runs/baselines_<timestamp>.json \
    --output-dir results/analysis/figures
```

This produces summary tables (CSV + Markdown) and ASR plots (overall, per-round,
per-technique, per-combination).

### Interactive / smoke tests

```bash
python test_app.py            # interactively query the sandbox target app
python test_llm_mutator.py    # verify the mutator produces non-trivial variants
```

---

## Tests

Test and smoke-test scripts live at the project root (`test_*.py`).
Those that require a running backend (e.g. `test_app.py`, `test_llm_mutator.py`)
will tell you if Ollama is unreachable or the model is missing. To run the
suite with `pytest`:

```bash
pip install -e ".[dev]"
pytest
```

---

## Ethics and responsible use

This project is **for academic security-evaluation research only**, against
local, sandboxed systems you control or are explicitly authorized to test.
See `configs/policy.yaml` for encoded ethical guidelines and constraints.

- Do **not** point this tool at systems without clear, documented authorization.
- Do **not** use it for harassment, fraud, unauthorized access, or any non-research goal.
- Unauthorized testing may violate computer-fraud and abuse laws in your jurisdiction.

The maintainers **disclaim all liability** for misuse. You assume full
responsibility for how you deploy and interpret results from this software.

## License

Released under the MIT License. See [LICENSE](LICENSE). Use is intended for
**academic and research purposes only**; the license does not grant permission
to use this software in ways that violate law or third-party rights.
