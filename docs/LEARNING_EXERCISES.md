# AI Engineering — Learning Exercises

A running list of hands-on experiments you can do with this repo to actually
*feel* how agentic RAG systems behave. Each exercise is self-contained — read
the "Why" first, then run the commands, then answer the reflection questions.

If you only do one thing on this page, do **Exercise 1**. It's the highest
learning-per-minute ratio on the whole list.

---

## Hardware notes (this machine)

- GPU: **NVIDIA RTX 3090, 24 GB VRAM**
- You can comfortably run any open-source model up to ~**32B parameters**
  at Ollama's default Q4 quantization.
- Rough VRAM math: `params (B) × 0.6 ≈ GB needed`, plus ~2 GB for context.
  - 7B ≈ 5 GB
  - 14B ≈ 9 GB
  - 32B ≈ 20 GB ← your practical ceiling for one model at a time
  - 70B ≈ 40 GB ← won't fit; will spill to CPU and crawl
- You have headroom to keep **two small models loaded simultaneously**
  (e.g. a 7B for agent orchestration + a 14B for final answers). See Exercise 3.

---

## Model shortlist worth pulling

```bash
# Small + fast (good for orchestration steps like plan/reflect/verify)
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull mistral:7b
docker exec ollama ollama pull phi3:mini

# Mid-tier (sweet spot for general Q&A on a 3090)
docker exec ollama ollama pull qwen2.5:14b
docker exec ollama ollama pull gemma2:9b

# Heavy hitters (best quality, still fits)
docker exec ollama ollama pull qwen2.5:32b
docker exec ollama ollama pull qwen2.5-coder:32b
docker exec ollama ollama pull deepseek-r1:32b          # "thinking" model
docker exec ollama ollama pull mistral-small:22b

# Check what you have
docker exec ollama ollama list
```

Ollama = the runner, not a model. Gemma, Qwen, Llama, Mistral, Phi, DeepSeek
etc. all run *through* Ollama. Switching "to Qwen" just means loading a
different file; the provider code in `backend/app/llm_provider.py` doesn't
change.

---

## Cost reminder

With `LLM_PROVIDER=ollama` (the default in `.env`), **every call is free and
local** — including the new agent loop, which makes 5–8 LLM calls per
question. "More calls" just means "slower," not "more money."

You only start paying when you set:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

Haiku is cheap enough that experimenting is pennies, but it's not zero. If you
want to compare a local model vs Claude, do it consciously and on a short
question set.

---

## Exercise 1 — Watch three models do the same job

**Why:** This is the most important exercise on the page. You'll see with your
own eyes that **model choice changes not just answer quality but whether the
agent loop even works reliably**. Small models frequently produce malformed
JSON in the reflect/verify steps and have to fall back. Big models don't. That
one observation is the core lesson of "why model choice matters for agents."

**Prereqs:**
```bash
make start-dev
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull qwen2.5:14b
docker exec ollama ollama pull qwen2.5:32b
```

**Run:**
```bash
python scripts/compare_models.py \
  -q "How do I perform a rolling update of a Kubernetes deployment and roll it back if it fails?" \
  -m llama3.1:8b qwen2.5:14b qwen2.5:32b \
  --save /tmp/compare.json
```

**What to look at in the output:**
1. **Latency** — how much slower is 32B than 8B? (Probably 3–5×.)
2. **`chunks kept` vs `dropped`** — does the small model mis-grade chunks?
3. **`retries`** — did any model have to rewrite and retry its query?
4. **`self-verified`** — did the model's own verify step say "yes this answers
   the question"? Does that match your own reading of the answer?
5. **Answer quality** — read all three answers. Which one actually helps?

**Reflection questions:**
- Was the best-answer model also the best-verified model?
- Did any model *claim* to have answered the question (`verified=true`) when
  it really hadn't? That's called "unfaithful self-verification" and it's a
  known failure mode.
- Is 32B's extra time worth the quality gap for your use case?

---

## Exercise 2 — DeepSeek-R1 and visible reasoning

**Why:** DeepSeek-R1 is a "reasoning model" — it thinks out loud *before*
answering, inside `<think>...</think>` tags. Running it through your agent
loop shows you two layers of orchestration at once: the agent's external loop
(plan/reflect/etc.) *and* the model's internal chain-of-thought. Great for
building intuition about what "reasoning" actually looks like.

**Prereqs:**
```bash
docker exec ollama ollama pull deepseek-r1:32b
```

**Run:**
```bash
python scripts/compare_models.py \
  -q "My pod is stuck in CrashLoopBackOff. Walk me through how to debug it." \
  -m qwen2.5:32b deepseek-r1:32b
```

**What to look at:**
- DeepSeek's answers will often contain visible `<think>` blocks. That's the
  model's internal reasoning.
- Does the agent's reflect step handle those tags well, or does it confuse
  the relevance grader?
- Is DeepSeek's final answer noticeably more structured than Qwen's?

---

## Exercise 3 — Mixed-size agent (orchestrator + answerer)

**Why:** In production, a common pattern is "use a cheap/fast model for the
boring orchestration steps, use the expensive/smart model for the one step
the user actually sees." Your 3090 can hold a 7B + a 14B simultaneously, so
you can try this locally.

**How (sketch, not prebuilt):**
1. In `backend/app/llm_provider.py`, add a second factory like
   `get_fast_provider()` that always returns Ollama with a small model
   (e.g. `llama3.1:8b`).
2. In `backend/app/agent.py`, wire PLAN / REFLECT / VERIFY to
   `get_fast_provider()` and leave GENERATE on the main
   `get_llm_provider()` (which can be a 32B model or Claude).
3. Add `OLLAMA_MAX_LOADED_MODELS=2` to `docker-compose.yml` under the
   `ollama` service so both stay resident.

**What you'll learn:**
- How much latency the orchestration steps actually contribute vs generation.
- Whether a small model is "smart enough" for structured-output tasks (JSON
  grading, yes/no verification) — often the answer is yes.
- The general pattern of "router model + worker model" that shows up in
  many agent frameworks.

---

## Exercise 4 — Real evaluation with RAGAS

**Why:** Right now the `/api/eval/run` endpoint falls back to a simple
Jaccard-overlap heuristic because RAGAS needs an LLM judge and defaults to
OpenAI. You have an Ollama model sitting right there — wire RAGAS to it and
you have a fully local, zero-cost evaluation pipeline.

**Steps:**
1. Read `backend/app/rag_eval.py` and find the fallback path.
2. Look up `ragas.llms.LangchainLLM` in the installed RAGAS version.
3. Use `langchain_community.llms.Ollama` to build a judge LLM pointed at
   `http://ollama:11434` with e.g. `qwen2.5:14b`.
4. Inject that judge into RAGAS before calling `ragas.evaluate(...)`.
5. Run the eval set:
   ```bash
   # You'll need to write this — see Exercise 5
   python scripts/run_eval_benchmark.py
   ```

**Reflection:**
- How do Ollama-judged scores compare to Jaccard-heuristic scores on the
  same answers?
- Is a small judge model (7B) enough, or does judging require a bigger one?
- What happens if you judge a model's answers with *itself* as the judge?
  (Spoiler: it cheats. That's why independent judges matter.)

---

## Exercise 5 — Benchmark script against the eval set

**Why:** Closes the loop between retrieval changes and measured quality. Right
now if you tweak `TOP_K_RESULTS` or swap embedding models you're guessing
whether it helped. Benchmark gives you numbers.

**Build `scripts/run_eval_benchmark.py` that:**
1. Reads `data/eval/devops_eval_set.json`.
2. For each sample, POSTs the `question` to `/api/agent/chat` and captures
   `answer`, `sources`, and `agent_trace`.
3. POSTs `{question, answer, contexts, ground_truth}` to `/api/eval/run`.
4. Tabulates results — per-topic averages for faithfulness, answer relevancy,
   context precision, context recall.
5. Saves a dated JSON report to `data/eval/reports/`.

**Then use it to measure real changes:**
- Run the benchmark with `TOP_K_RESULTS=3`, then `5`, then `10`. Does quality
  keep going up? Where's the knee?
- Run it with `HYDE_ENABLED=true` vs `false`. Does HyDE actually help on your
  doc set?
- Run it with different embedding models (requires re-ingestion). Is
  `BAAI/bge-base-en-v1.5` really the best choice, or is `bge-large` worth the
  extra VRAM?

---

## Exercise 6 — Stream the agent trace as SSE

**Why:** Right now `/api/agent/chat` makes the user wait for the entire
plan→retrieve→reflect→decide→generate→verify cycle before returning anything.
In real agent UIs (Cursor, Claude Code, etc.), you see each step surface as it
happens. Building that teaches you async generators, Server-Sent Events, and
how to surface intermediate reasoning without exposing raw model internals.

**Sketch:**
1. Add `POST /api/agent/chat/stream` in `main.py` returning
   `StreamingResponse(..., media_type="text/event-stream")`.
2. In `agent.py`, convert `run()` into an async generator `run_stream()` that
   `yield`s each trace step as a dict the moment it completes.
3. Serialize each yield as an SSE event: `f"event: {step}\ndata: {json}\n\n"`.
4. When GENERATE starts, stream the answer tokens themselves as a separate
   event type.
5. Write a small curl test:
   ```bash
   curl --no-buffer -N -X POST http://localhost:8000/api/agent/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain Terraform state locking"}'
   ```

---

## Exercise 7 — Swap embedding models and watch retrieval change

**Why:** The retriever is upstream of everything. A better embedding means
better chunks reach the LLM, which means better answers regardless of which
LLM you use. Worth feeling firsthand how much this matters.

**Models to try (all run on CPU fine, or flip to GPU in `vectorstore.py`):**
- `BAAI/bge-base-en-v1.5` — current default, 768-dim
- `BAAI/bge-large-en-v1.5` — 1024-dim, better quality
- `intfloat/e5-large-v2` — different architecture, often competitive
- `nomic-ai/nomic-embed-text-v1.5` — newer, strong on technical docs

**Steps:**
1. Set `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` in `.env` to match.
2. `make ingest` — FULL re-index is required when you change embeddings
   (chunks get different vectors). This takes a while.
3. Run your benchmark from Exercise 5 before and after. Compare.

---

## Exercise 8 — Compare Ollama vs Claude Haiku on the same eval set

**Why:** You built a provider abstraction — use it. This is the "is paying for
an API worth it" question answered empirically for *your* documents.

**Steps:**
1. Get an Anthropic API key. Set `ANTHROPIC_API_KEY=...` in `.env`.
2. Run benchmark (Exercise 5) with `LLM_PROVIDER=ollama` and
   `OLLAMA_MODEL=qwen2.5:32b`. Save report.
3. Run it again with `LLM_PROVIDER=anthropic` and
   `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`. Save report.
4. Diff the reports. Also note total $$ spent (Anthropic returns token counts
   per call — log them).

**What you'll learn:**
- For a specific task (DevOps Q&A on your docs), is a 32B local model
  competitive with a frontier cloud model?
- Where does Claude win? Where is the local model good enough?
- What's the cost of "just use Claude for everything" on a realistic workload?

---

## Tracking what you've tried

Keep a log — even just a markdown file — of each experiment, the config, and
the numbers you saw. AI engineering is mostly about noticing small differences
between configurations, and you won't notice them if you don't write them
down.

Suggested format:

```
## 2026-04-14 — Exercise 1
- Question: K8s rolling update rollback
- Models: llama3.1:8b, qwen2.5:14b, qwen2.5:32b
- Notable: 8b failed reflect JSON twice, triggered retry; 32b self-verified
  correctly; 14b answer was actually the most concise and readable
- Takeaway: 14b is the sweet spot for this type of question on a 3090
```
