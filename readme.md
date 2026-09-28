# Autonomous arXiv Paper Digest & QA Agent

Takes a topic or arXiv ID, retrieves and parses the paper, produces a structured executive briefing, and answers follow-up questions grounded in the paper's actual text (RAG).

## Architecture

Explicit state graph built with LangGraph. One shared `AgentState` (see `state.py`) flows through every node; each node returns only the fields it changed.

```
understand → retrieve → select → fetch → chunk → summarize → END
                │           │        │
            no_results  ambiguous  parse_failed
                │           │        │
                └────────→ END ←─────┘
```

QA is intentionally **not** part of the compiled graph. It's a repeated, interactive step, not a one-shot pipeline stage — `qa_node()` is called directly in a loop from `main.py` against the same session state, so the vector store and conversation history persist across turns without re-entering the graph each time.

**Nodes**
| Node | Does | Writes to state |
|---|---|---|
| `query_understanding` | Regex-detects arXiv ID/URL vs. free-text topic | `intent` |
| `arxiv_retrieval` | Calls the arXiv Atom API | `candidates`, `status` |
| `selection_ranking` | Cosine-similarity ranks abstracts against the query; flags ambiguous topics instead of guessing | `selected_paper` or `needs_disambiguation` |
| `fetch_parse` | Downloads PDF, extracts text via PyMuPDF, splits into sections by heading regex; falls back to abstract if PDF fails; detects non-English papers | `raw_text`, `parsed_sections` |
| `chunk_embed` | Fixed-size overlapping chunks → embeddings → local Chroma collection; handles short papers as single chunk | `chunks`, `vector_store_ref` |
| `summarize` | Map-reduce summarization for large papers; plain-text LLM call; `limitations` never left empty | `briefing` |
| `qa_node` (outside graph) | Embed question → retrieve top-6 chunks → deduplicate → answer or refuse | `qa_history` |

**State shape**: see `state.py`. Full `AgentState` TypedDict — paper metadata, parsed text, chunk list, vector store handle, the generated briefing, and `qa_history` (list of Q/A/grounding-flag turns) all live in one object for the session.

## Setup & Run

**Step 1: Clone the repo and create a virtual environment**

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Mac/Linux:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Get a free API key from OpenRouter**

1. Go to https://openrouter.ai
2. Sign up and create an API key
3. Set the environment variable:

Windows (PowerShell):
```powershell
$env:OPENROUTER_API_KEY="your_key_here"
```

Mac/Linux:
```bash
export OPENROUTER_API_KEY=your_key_here
```

**Step 3: Run**

```bash
# With a specific arXiv ID
python main.py 2609.30264

# With a natural-language topic
python main.py "recent work on KV-cache compression for LLMs"

# With a full arXiv URL
python main.py https://arxiv.org/abs/2609.30264
```

**Rate limits**: OpenRouter's free tier uses `openrouter/auto` which routes to available free models. The pipeline makes one LLM call for summarization (map-reduce for large papers) plus one per QA question. No rate limit issues expected for single sessions.

**Embeddings**: run locally via `sentence-transformers` (all-MiniLM-L6-v2, CPU, no API calls, no rate limits).

## Example Run

### Paper 1: AD-WM (2609.30264)

```
$ python main.py 2609.30264

Downloading from: https://arxiv.org/pdf/2609.30264v2
Downloaded to: C:\Users\...\Temp\2609.30264v2.pdf
Extracted 41848 characters
Large paper detected, using map-reduce summarization...
Summarizing section 1/6...
Summarizing section 2/6...
Summarizing section 3/6...
Summarizing section 4/6...
Summarizing section 5/6...
Summarizing section 6/6...

======================================================================
AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control
Jiabin Qiu, Zixuan Chen, Hongye Cao, Jieqi Shi, Jing Huo, Yang Gao
arXiv:2609.30264v2  (2026-09-24)  https://arxiv.org/abs/2609.30264v2
======================================================================

Why it matters:
  World models for MPC are often trained for factual prediction accuracy, but planning requires
  counterfactual action discrimination: the ability to tell apart the outcomes of different candidate
  action sequences. AD-WM shows that explicitly preserving action-dependent differences in latent
  dynamics — rather than maximizing raw prediction accuracy — can improve action ranking, closed-loop
  control, and even zero-shot real-robot transfer. This suggests a guiding principle for learned world
  models in control: planning performance should be the objective, not prediction fidelity alone.

Problem:
  Standard factual predictors in JEPA-based world models can achieve high predictive accuracy while
  failing to preserve action-dependent information that is critical for counterfactual action selection,
  causing the world model to rank candidate actions incorrectly and degrade MPC performance.

Method:
  - AD-WM combines residual latent prediction with predictor-level action-recovery regularization:
    auxiliary inverse dynamics and normalized action-recovery heads reconstruct action embeddings
    from current and predicted latent states.
  - These auxiliary objectives shape the learned latent transitions so that counterfactual action
    sequences produce distinguishable outcomes, while the residual dynamics supervise actual
    successor representations.
  - At deployment, the auxiliary heads are discarded and the learned predictor is used with CEM-MPC,
    which evaluates candidate actions via terminal cost against a goal representation.
  - The paper introduces planning-aligned diagnostics: counterfactual action discriminability
    (Spearman rank correlation between predicted and true costs) and elite-set regret.
  - Evaluations include multiple simulation benchmarks with hard-start protocols, ablations,
    and zero-shot transfer to a real Franka robot.

Key results:
  - AD-WM outperforms a matched LeWM baseline across all Cube protocols and achieves the highest
    success under the largest perturbation.
  - AD-WM improves hard-start success from 3.7% to 52.0% over a matched LeWM baseline.
  - Zero-shot transfer to a real Franka robot: basic pick-and-place from 42.2% to 71.1%.
  - Ablations identify residual prediction, inverse modeling, and MI regularization as key components.
  - The proposed CAD and elite-regret diagnostics reveal a mismatch between factual prediction
    accuracy and effective action selection.

Limitations:
  - AD-WM does not dominate every setting: INTACT Direct performs better under smaller perturbations.
  - The new diagnostics require access to true costs, better suited for offline evaluation than online.
  - Auxiliary action-recovery heads are discarded at deployment; performance may be sensitive to
    the weighting of the auxiliary and mutual-information losses.

Suggested follow-up questions:
  - What datasets and environments were used for evaluation?
  - What baselines did they compare against and what were the results?
  - What did the ablation studies show about individual components?

Ask questions about the paper (blank line to quit).

> What datasets were used for evaluation?

The excerpts mention the filtered cadene/droid 1.0.1 data [34] for the transfer evaluation
on the Franka setup. They also refer to "Cube" as the setting for the metrics in Table III,
and factual-error evaluation uses 61,999 held-out clips with three-frame context.
No other dataset names are given.

[grounded in chunks: a74e9e42, b1cd5eed, e57c19af, 90820e1b, f3a12c91, d8b23e45]

> How does this compare to prior world model approaches?

Based on the excerpts, AD-WM is contrasted with prior JEPA world models: prior approaches can
achieve low prediction error by largely preserving the current representation, but this can
obscure differences between candidate actions. AD-WM instead preserves action-dependent changes
to guide goal-directed selection. AD-WM improves hard-start success from 3.7% to 52.0% over a
matched LeWM baseline and improves mean success over the reproduced baseline in four of five
simulation environments. It also improves zero-shot transfer to a Franka setup, increasing
basic pick-and-place success from 42.2% to 71.1% without lab-specific adaptation.

[grounded in chunks: d881a1a9, 3b260b0a, 923513a5, a6bf76e5, 5fc9d8df, 7568f922]

> What are the computational requirements?

The excerpts do not explicitly state computational requirements such as hardware, memory,
training time, or total compute. They only mention some related details like batch size 8
per GPU and 800 CEM samples, 10 elites, and 10 iterations.

[grounded in chunks: e0bd2685, ab017c27, c3a465d8, b51ff1e4, 1366f238, b390ef15]

> What loss functions or objectives do they use?

The paper uses a combined objective with weighted loss terms: igLsig + λinvLinv + λMILMI (Eq. 5),
where Lsig is the SIGReg representation regularizer (same as LeWM), Linv is an inverse dynamics
recovery objective, and LMI is a normalized recovery objective motivated by conditional mutual
information. Prediction supervision is also used, with Lpred = ∥ˆzt+1 − zt+1∥₂² (Eq. 2).

[grounded in chunks: a0370f9e, fa7cabad, 6f2c8f28, 19c5fde9, 173aac30, 0c7350dc]

> What are the success rates on different tasks?

The reported success rates are:
- TwoRoom: 90% (LeWM) vs 98% (AD-WM)
- Reacher: 76.7% (LeWM) vs 83.3% (AD-WM)
- Push-T: 94% (LeWM) vs 92% (AD-WM)
- OGBench-Cube: 73.3% (LeWM) vs 90.7% (AD-WM)
- Scene overall: 35.5% (LeWM) vs 39.5% (AD-WM)
- Scene subtasks — Drawer: 63.3% vs 69.3%, Window: 46.7% vs 55.3%
- Real-robot basic pick-and-place: 42.2% → 71.1% overall;
  per-object: 33.3%→60.0%, 46.7%→73.3%, 46.7%→80.0%

[grounded in chunks: 3a927ea2, 387e8ca7, 5e0d82b2, 0241b520, 3243bcc7, 173aac30]

> How does this compare to GPT-4?

The paper doesn't cover this; the excerpts contain no comparison to GPT-4.

[grounded in chunks: 99cbaf08, 19c5fde9, 173aac30, eae8e5a1, 0241b520, 4901e85a]
```

---

### Paper 2: Attention Is All You Need (1706.03762)

```
$ python main.py 1706.03762

======================================================================
Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
arXiv:1706.03762v7  (2017-06-12)  https://arxiv.org/abs/1706.03762v7
======================================================================

Why it matters:
  The Transformer introduces a sequence-transduction architecture that entirely replaces recurrence
  and convolution with attention mechanisms, overcoming the sequential-computation bottleneck of
  traditional RNNs. This enables unprecedented parallelization, faster training, and state-of-the-art
  performance on machine translation while also generalizing to other tasks like parsing. Its design
  principles — self-attention, multi-head attention, and positional encodings — have become
  foundational for modern large language models.

Problem:
  Traditional sequence models rely on recurrence or convolution, which limit parallelization and
  make it computationally expensive to capture long-range dependencies. Recurrent networks process
  tokens sequentially, creating a bottleneck for training on long sequences.

Method:
  - Architecture consists of stacked multi-head self-attention and feed-forward layers in both
    encoder and decoder, with residual connections and layer normalization.
  - Scaled dot-product attention: queries and keys multiplied, divided by √dₖ, softmax to weight values.
  - Multi-head attention runs 8 parallel attention operations (each of dimension 64) in three roles:
    encoder-decoder attention, encoder self-attention, and masked decoder self-attention.
  - Positional encodings (sinusoidal) added to input embeddings to represent sequence order.
  - Trained with Adam optimizer, warmup-based learning-rate schedule, dropout, label smoothing,
    checkpoint averaging and beam search during inference.

Key results:
  - BLEU scores of 28.4 on English-to-German and 41.0 on English-to-French translation.
  - Outperformed all previous models at a fraction of the training cost.
  - Generalizes to English constituency parsing.
  - Training done on 8 P100 GPUs.

Limitations:
  - Self-attention has quadratic complexity with respect to sequence length.
  - Sinusoidal positional encodings may have limits for very long sequences.
  - Generalizability to other domains (speech, images) not demonstrated in the paper.

Suggested follow-up questions:
  - What BLEU scores did the Transformer achieve on translation tasks?
  - How many attention heads were used and what were the key hyperparameters?
  - How did the base model compare to the large model?

Ask questions about the paper (blank line to quit).

> What BLEU scores did the Transformer achieve on translation tasks?

The Transformer achieved BLEU scores of 28.4 on English-to-German and 41.0 on
English-to-French translation, outperforming all previous models at lower training cost.

[grounded in chunks: f2213a21, 4a4e06b9, ed21c0d3, dbeeb230, 9954e2f9, 36275709]

> How many attention heads were used and what were the key hyperparameters?

The paper uses 8 attention heads with dk = dv = dmodel/h = 64. Dot products are scaled
by 1/√dk. The model was trained with Adam optimizer with β1=0.9, β2=0.98, ε=10^-9,
with a warmup learning rate schedule over 4000 steps. Dropout rate of 0.1 and label
smoothing of ε_ls=0.1 were used.

[grounded in chunks: 43c7874d, cf8d70b7, a3f6ea93, 7d697b2f, 4415e2ac, f6aebc92]

> Did they compare against GPT-4?

The paper doesn't cover this; GPT-4 did not exist when this paper was written in 2017.

[grounded in chunks: 840384ed, 6fac4a00, 4edabb4f, f2213a21, ed21c0d3, a3f6ea93]
```

## Design Decisions & Tradeoffs

**Chunking**: fixed-size character chunks (1000 chars, 300 overlap) rather than semantic/section-aware chunking. Simple, predictable, and good enough for a single paper. Higher overlap (300 vs default 150) reduces the chance of relevant content being split across chunk boundaries. With more time I'd chunk per-section using the regex-detected headings so retrieval could be filtered by section.

**Ranking without an LLM call**: topic-search ranking uses embedding cosine-similarity against abstracts, not an LLM judgment call. Cheaper and deterministic. The `top_score - second_score < 0.03` ambiguity threshold is a heuristic — not tuned against real query data but works well for clear topics.

**Map-reduce summarization**: papers over 12,000 characters are split into 4,000-character sections, each summarized separately, then combined into a final briefing. This handles large papers without truncating the second half entirely — the biggest practical failure mode for a fixed-excerpt approach.

**Grounding**: the LLM is explicitly instructed to answer only from retrieved chunks and say "not covered" if the answer isn't there. Chunk IDs are shown alongside every answer so grounding is verifiable. The similarity gate was removed in favour of always retrieving top-6 chunks and relying on prompt-level grounding — more reliable given embedding distance inconsistencies with Chroma's default metric.

**6 chunks per QA query**: increased from the default 4 to improve coverage. Duplicate chunks are deduplicated before sending to the LLM to avoid over-emphasising repeated content from overlapping windows.

**Abstract fallback**: if the PDF download or parse fails, the pipeline falls back to the abstract text from arXiv metadata rather than crashing. The briefing will be shallower but the pipeline still completes.

**Language detection**: non-English papers are flagged with a warning rather than silently producing degraded output. The pipeline continues but the user is informed.

**State persistence**: everything lives in-memory in `AgentState` for the process lifetime — no DB, no session store. Fine for a CLI single-session tool. Would need SQLite + serialized Chroma paths to resume sessions.

**LLM provider**: OpenRouter's `openrouter/auto` routes to available free models automatically, avoiding the model deprecation issues that affect hardcoded model names on Groq and Gemini.

**What I'd do with more time**:
- Section-aware chunking + retrieval filtering by section
- Tune chunk size and overlap per paper length
- Persist sessions to disk so QA can resume without re-parsing
- Use `pdfplumber` instead of PyMuPDF for better table extraction
- Add `unstructured.io` for layout-aware parsing of complex PDFs
- Tune grounding similarity threshold against labelled in-paper vs out-of-paper questions

**Known limitations**:
- Figures, charts, and equations extract poorly from PDF text layer
- Very short papers (workshop abstracts) produce few chunks, limiting QA quality
- Session state is lost if the process exits — no resume capability
- OpenRouter free tier has rate limits; rapid QA sessions may hit them

## Failure Cases Handled

| Failure | How handled |
|---|---|
| Zero arXiv results | Clear error message + suggestion to broaden topic |
| Many ambiguous results | Shows top 5 candidates with IDs, asks user to pick |
| PDF download fails | Falls back to abstract text from arXiv metadata |
| Scanned/image PDF | Detected via <500 char extraction; clear error message |
| Huge paper | Map-reduce: summarize sections separately, combine |
| Short paper | Full text used as single chunk, no splitting |
| Non-English paper | Warning printed, pipeline continues |
| Duplicate chunks | Deduplicated before sending to LLM |

## Out of Scope (per assessment)
No UI beyond CLI, no auth/deployment, no non-arXiv sources, no fine-tuning.