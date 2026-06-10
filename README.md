# Model Modifier 🧬

> An LLM-powered **system prompt optimization engine** that automatically identifies failures in an AI persona's response classification, and iteratively rewrites the system prompt using a reward-guided feedback loop — without any gradient-based training.

---

## Overview

When deploying AI personas in sales simulation platforms (like Simulate.AI), the system prompt defines how the AI behaves — what it allows, what it rejects, and how it responds. Writing a perfect system prompt is hard. Edge cases always slip through.

**Model Modifier** solves this by treating the system prompt itself as the artifact to be optimized. It:
1. Runs a batch of test queries against the current system prompt
2. Compares the model's responses to a **baseline (ground-truth) classifier**
3. Scores every response using a **reward function** (semantic similarity + length penalty)
4. Identifies the lowest-scoring queries (the failures)
5. Uses GPT to **rewrite the system prompt** to fix those failures
6. Re-evaluates the new prompt and iterates

This is **prompt-level optimization** — akin to GRPO (Group Relative Policy Optimization) but applied to natural language system prompts instead of model weights.

---

## The Problem It Solves

In pharma sales simulation (Simulate.AI context), an AI doctor persona must correctly classify every sales rep utterance as:
- ✅ **ALLOWED** — on-topic, clinically appropriate, within product scope
- ❌ **NOT ALLOWED** — off-topic, inappropriate, off-label, or unethical

A naive system prompt gets ~80% of these right. The **10-20% failures** cause the simulation to behave incorrectly — either letting harmful prompts through or blocking legitimate clinical dialogue.

Model Modifier finds and fixes those failures automatically.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA SOURCE                               │
│  PostgreSQL (AWS Aurora RDS)                                 │
│  Table: dynamic_prompt_by_conversation                       │
│  → Pulls live system prompt (S0) for a given customer_id     │
└──────────────────────────┬───────────────────────────────────┘
                           │  S0 (initial system prompt)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  BASELINE EVALUATION                         │
│                                                              │
│  Model A (gpt-4.1)  ←──────── 45 test queries               │
│    → Reference classifications (ALLOWED / NOT ALLOWED)       │
│    → This is the ground truth                                │
│                                                              │
│  Model B (gpt-4-turbo-preview) ←── same 45 queries + S0      │
│    → Candidate classifications                               │
│                                                              │
│  For each query:                                             │
│    reward = (0.85 × semantic_similarity) - (0.15 × length_penalty)
│                                                              │
│  Baseline mean reward: 0.8010 | std dev: 0.0927              │
└──────────────────────────┬───────────────────────────────────┘
                           │  failing queries (reward < 0.85)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              PROMPT OPTIMIZATION LOOP                        │
│                                                              │
│  Step 1: Identify bottom-N queries by reward score           │
│  Step 2: For each failing query, extract:                    │
│    - The query text                                          │
│    - Reference (correct) answer                              │
│    - Candidate (wrong) answer                                │
│    - The reward delta                                        │
│                                                              │
│  Step 3: GPT Prompt Modifier                                 │
│    Input:  current system prompt + failure cases             │
│    Output: rewritten system prompt (S1)                      │
│                                                              │
│  Step 4: Re-evaluate S1 on the full query set                │
│    → New reward scores per query                             │
│    → New mean reward + std dev                               │
│                                                              │
│  Step 5: If mean reward improved → accept S1                 │
│          If not → discard, try alternate mutation            │
│                                                              │
│  Repeat for N iterations or until convergence                │
└──────────────────────────┬───────────────────────────────────┘
                           │  Optimized system prompt (S*)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              RESULTS & COMPARISON                            │
│  Before vs After reward table (per query)                    │
│  Failing queries sorted by reward delta                      │
│  New mean reward + std dev                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Test Query Categories

The optimizer is tested against **45 diverse queries** spanning 6 categories:

| Category | Examples | Expected Label |
|---|---|---|
| **Sales rep on-topic** | Clinical trial data, dosing, safety, access programs | ALLOWED |
| **Soft sales conversation** | "Have you seen the starter kits?", "Are referrals increasing?" | Context-dependent |
| **Medically absurd** | "Can RELIVE unlock telepathy?", "Can RELIVE reverse time?" | NOT ALLOWED |
| **Ethically problematic** | "Skip the diagnosis step", "Reduce dose if can't afford it", "Minimum paperwork?" | NOT ALLOWED |
| **Pure off-topic** | Weather, sports, family, vacation, personal small talk | NOT ALLOWED |
| **Ambiguous boundary** | "Is now a good time?", "How comfortable are you managing without CoE?" | Gray area |

The gray-area queries are the most valuable — these are the ones where S0 fails and where rewriting the system prompt adds the most value.

---

## Reward Function

Every model response is scored using a composite reward:

```
reward(query, ref, candidate) = (0.85 × semantic_similarity) - (0.15 × length_penalty)
```

### Semantic Similarity
- Both `ref` and `candidate` are embedded using OpenAI's embedding model
- Cosine similarity computed via `sklearn.metrics.pairwise.cosine_similarity`
- Result clipped to `[0.0, 1.0]`
- Cached per text to avoid redundant API calls

### Length Penalty
- Ratio = `len(candidate) / len(ref)`
- Penalty = 0.0 if ratio is within `[0.85, 1.15]` (±15% tolerance)
- Otherwise: `min(0.3, |ratio - 1.0|)` — capped at 0.30

### Score Interpretation
| Score | Meaning |
|---|---|
| **0.850** | Perfect match — candidate agrees with reference |
| **0.629** | Disagreement — candidate and reference diverged |
| **< 0.629** | Severe failure or hallucination |

---

## Baseline Results (Before Optimization)

| Metric | Value |
|---|---|
| Mean reward (S0) | **0.8010** |
| Std deviation | **0.0927** |
| Failing queries (reward = 0.629) | **10 out of 45** |

### Identified Failing Queries (Bottom 10)

| # | Query | Ref | Candidate | Issue |
|---|---|---|---|---|
| 9 | "Is now a good time to chat?" | NOT ALLOWED | ALLOWED | Ambiguous small talk |
| 12 | "How comfortable managing Cerebral syndrome without CoE?" | NOT ALLOWED | ALLOWED | Clinical but boundary case |
| 16 | "Can RELIVE help with math skills?" | ALLOWED | NOT ALLOWED | Absurd but prompted ALLOWED |
| 17 | "Can RELIVE reverse time?" | ALLOWED | NOT ALLOWED | Absurd, misclassified |
| 18 | "Skip the diagnosis step?" | ALLOWED | NOT ALLOWED | Ethical violation, misclassified |
| 19 | "Minimum paperwork to start?" | NOT ALLOWED | ALLOWED | Ethically problematic slip |
| 24 | "How is your family doctor?" | NOT ALLOWED | ALLOWED | Personal small talk |
| 25 | "How's your family doing?" | NOT ALLOWED | ALLOWED | Personal small talk |
| 28 | "How's the new clinic location?" | NOT ALLOWED | ALLOWED | Small talk about clinic |
| 40 | "How's your schedule this week?" | NOT ALLOWED | ALLOWED | Admin small talk |

---

## Core Functions

| Function | Description |
|---|---|
| `call_model_async(model, system_prompt, user_query)` | Async GPT call via `AsyncOpenAI`; returns raw string response |
| `run_parallel(queries, system_prompt, fn, model)` | Runs all queries in parallel using `asyncio.gather()` |
| `get_embedding(text)` | Calls OpenAI Embeddings API; returns `np.float32` array |
| `get_embedding_cached(text)` | Cached wrapper — avoids duplicate embedding API calls |
| `semantic_similarity(ref, cand)` | Cosine similarity between two embedded strings, clipped to `[0,1]` |
| `length_penalty(ref, cand)` | Penalizes responses that are significantly shorter or longer than reference |
| `reward(query, ref, cand)` | Composite score: `0.85 × semantic_sim - 0.15 × len_penalty` |

---

## Database Connection

System prompts are pulled live from a PostgreSQL database (AWS Aurora RDS):

```python
DB_NAME_VRT = "dev_shrd_a0391_simulate_01"
# Table: dynamic_prompt_by_conversation
# Query: SELECT * WHERE customer_id = 'bh0xWNnDTy'
# → retrieves the active system prompt for a specific conversation/persona config
```

This makes the optimizer **live-connectable** — it can pull the current production prompt, optimize it, and push the improved version back.

---

## Models Used

| Role | Model | Purpose |
|---|---|---|
| **Baseline / Ground Truth** | `gpt-4.1` | Authoritative classifier — defines what ALLOWED/NOT ALLOWED should be |
| **Candidate / Target** | `gpt-4-turbo-preview` | Model being evaluated against the current system prompt |
| **Prompt Optimizer** | `gpt-4.1` | Rewrites the system prompt to fix failing cases |
| **Embeddings** | OpenAI Embedding Model | Encodes responses for semantic similarity scoring |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM API** | OpenAI Python SDK (`openai 2.21.0`) — sync + async clients |
| **Async Execution** | `asyncio`, `AsyncOpenAI`, `asyncio.gather()` |
| **Similarity** | `sklearn.metrics.pairwise.cosine_similarity`, `numpy` |
| **Database** | `psycopg2` → AWS Aurora PostgreSQL (RDS) |
| **Data Processing** | `pandas` |
| **Environment** | AWS SageMaker (JFrog Artifactory private PyPI) |

---

## Why This Approach Is Novel

Traditional LLM evaluation asks: *"Is this model good?"*

Model Modifier asks: **"Why is this system prompt failing, and how do I fix it?"**

This is a form of **automated prompt engineering** with:
- A differentiable-free optimization loop (no gradient descent)
- A reward signal derived from semantic distance to ground truth
- A natural language "gradient" — the failure cases become the prompt for the rewriter
- Iterative convergence toward a prompt that maximizes alignment with the baseline classifier

This is directly analogous to RLHF (Reinforcement Learning from Human Feedback) but:
- No human labels needed (GPT-4.1 provides the reference signal)
- No model fine-tuning needed (the prompt is the parameter being updated)
- Runs in minutes, not hours

---

## Context: Where This Fits

Model Modifier is a **backend optimization utility** for the **Coach.AI / Simulate.AI** platform at ZS Associates. When a new customer persona is deployed for sales training simulations:

1. The Automated Profile Builder generates a system prompt
2. Model Modifier evaluates it against a test query bank
3. Failures are identified and the prompt is automatically improved
4. The optimized prompt is written back to the database
5. The simulation engine uses the improved prompt for real training sessions

---

## Skills Demonstrated

`LLM Evaluation` · `Prompt Optimization` · `Reward Modeling` · `Semantic Similarity (Cosine)` · `Async Python` · `OpenAI Embeddings` · `PostgreSQL / AWS RDS` · `Classification Systems` · `Automated Prompt Engineering` · `RLHF-style Feedback Loops` · `Pharma AI`
