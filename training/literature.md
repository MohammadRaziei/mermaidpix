# literature.md — does prior research exist on image → Mermaid (or the adjacent problem)?

Written in response to: "is anyone in the literature actually working on
png → mmd, or is this untouched." Short answer: **one directly on-topic
paper exists (Dec 2025), and it takes a completely different approach from
this project** (prompting a frontier VLM, not training a small dedicated
model) — so this project isn't duplicating existing work, it's exploring a
different point in the design space of the same problem. Beyond that one
paper, there's a substantial adjacent literature on flowchart/diagram
image understanding and on image-to-markup generation more broadly, which
this project's architecture (TrOCR-style ViT encoder + small transformer
decoder, see `IDEA.md`) sits squarely within lineage-wise.

## 1. Directly on-topic: image → Mermaid specifically

**Flowchart2Mermaid (Deka & Devereux, Queen's University Belfast, Dec
2025, [arXiv:2512.02170](https://arxiv.org/abs/2512.02170))** — the one
paper found that does exactly this task. A web app: upload a flowchart
image, a *prompted* frontier VLM (GPT-4.1, GPT-4.1-mini, GPT-4o,
GPT-4o-mini, or Gemini-2.5-Flash — no fine-tuning, just a carefully
engineered system prompt) produces Mermaid code, which the user then
refines through inline editing, drag-and-drop node insertion, or
natural-language commands. Evaluated on a 200-image subset of the FlowVQA
dataset (below) with two metric families:
- **Symbolic**: precision/recall/F1 on extracted nodes and directed edges
  (via an LLM-judge that normalizes IDs/shapes/quoting before comparing),
  plus SBERT cosine similarity between predicted and gold Mermaid text.
- **High-level structural** (also LLM-judged): structural accuracy, flow
  accuracy, syntax validity, semantic fidelity, completeness.

Results: the strongest models (Gemini-2.5-Flash, GPT-4.1-mini) hit
node/edge F1 in the high 0.98s and near-1.0 on the structural metrics;
weaker models (GPT-4o-mini) have near-perfect syntax validity but
noticeably worse structural/flow scores — i.e. **syntactically valid
Mermaid that's still structurally wrong is a real, observed failure mode
in this literature**, not just a hypothetical concern.

**Why this project differs, not duplicates:** Flowchart2Mermaid's whole
approach is "call a frontier VLM API with a good prompt" — no training,
no small dedicated model, no offline/local inference story. This
project's explicit constraint (stated at the start of this work) was the
opposite: a compact, locally-runnable, purpose-trained model (~105M
params, see `IDEA.md`'s size analysis) instead of routing every inference
through GPT-4.1/Gemini. Nobody in the literature found so far has done
*that* specifically for Mermaid — trained a small dedicated image-to-code
model for this exact target syntax. That's this project's actual
contribution/gap, not something already covered.

**Their evaluation metrics are worth borrowing.** This project currently
only measures token-level accuracy and exact-string-match (see
`train_reconstructor.py`'s `qualitative_samples`). Flowchart2Mermaid's
node/edge-level precision/recall/F1 (ignoring node IDs, shape brackets,
and quote style — i.e. comparing *content*, not exact syntax) is a much
more forgiving and more informative metric: two Mermaid programs that
differ only in variable-naming or bracket style are structurally
identical but would currently score as a total miss under exact-match.
Worth adopting a similar node/edge-set comparison as an additional metric
once training is validated.

**FlowVQA (Singh et al. 2024)** — the dataset Flowchart2Mermaid evaluates
against. Real-world (not synthetically rendered) flowchart images with
step-level semantic annotations, originally built for visual-question-
answering, repurposed here because its annotations happen to include
Mermaid representations. Relevant as a candidate **out-of-distribution
eval set**: since every image this project trains and validates on comes
from `mermaidx`'s own renderer (see the pixel-comparison-determinism
discussion in `IDEA.md`), FlowVQA's real-world images would be a genuinely
independent check of whether the model generalizes beyond mermaidx's
particular visual style, fonts, and layout conventions -- something no
amount of on-the-fly mermaidx-rendered training data can tell you on its
own.

**MermaidSeqBench ([arXiv:2511.14967](https://arxiv.org/abs/2511.14967),
Nov 2025)** — evaluates LLM-generated Mermaid *sequence* diagrams, but
from **natural-language descriptions**, not images (NL → Mermaid, not
image → Mermaid). Different modality, not directly applicable, but its
LLM-as-judge methodology (fine-grained metrics: syntax correctness,
activation handling, error handling, practical usability) is another
example of the same "don't rely on exact string match" pattern seen in
Flowchart2Mermaid above.

**Commercial tools** (ChatFlowchart, ImageToMermaid.com, dAIgram,
FlowchartAI, DiagramGPT) — all found to be thin wrappers around GPT-4V/
similar VLM APIs, same category as Flowchart2Mermaid's approach but
without any published methodology or evaluation. Not research, listed
here only to confirm the "image to Mermaid" product space is VLM-prompt
based across the board, not just in the one paper found.

## 2. Adjacent: flowchart/diagram image → structured representation (not necessarily Mermaid)

A longer-running literature exists on parsing diagram *structure*
(nodes, edges, shapes) from images, mostly predating the VLM-prompting
approach and mostly NOT producing directly-executable/renderable code as
output:

- **GRCNN ([arXiv:2011.05980](https://arxiv.org/abs/2011.05980), 2020)**
  — end-to-end CNN that predicts flowchart node and edge information
  directly from the image, then synthesizes a program from the recovered
  graph. Reports 94.1%/67.9% edge/node accuracy and 66.4% end-to-end
  program-synthesis accuracy on their own dataset. Closest in spirit to
  a from-scratch trained model (like this project), but predicts a graph
  structure + separately synthesizes code, rather than directly
  generating target markup text end-to-end the way this project's
  decoder does.

- **Arrow R-CNN (Schäfer & Stuckenschmidt, referenced in multiple papers
  above) / DrawnNet (Fang et al. 2022)** — extend Faster R-CNN with
  arrow head/tail keypoint detection for hand-drawn diagrams: detect
  symbols and arrows as objects, then assemble structure in a
  post-processing step. This is the **"detect-then-assemble" paradigm**,
  architecturally the opposite of this project's **end-to-end
  sequence-generation paradigm** (encoder → decoder emits the target
  syntax token-by-token, no separate object-detection or assembly step).
  Worth naming explicitly as the road not taken here, and a real
  alternative approach if the current architecture's accuracy plateaus.

- **Flowmind2Digital ([arXiv:2401.03742](https://arxiv.org/abs/2401.03742),
  2024)** — most comprehensive recent hand-drawn flowchart/mind-map
  recognition system, converts to PPT/Visio (not Mermaid). Introduces the
  hdFlowmind dataset (1,776 hand-drawn diagrams, 22 scenarios) and reports
  87.3% accuracy, +11.9pp over prior methods. Notably: their ablation
  found **simplifying the graphics improved accuracy by 9.3%** — a
  finding that cuts directly against adding heavy visual noise/complexity
  during training, and is worth keeping in mind if this project's own
  augmentation (see `train_reconstructor.py`'s rotation/color-jitter) is
  ever tuned more aggressively — the literature's own evidence points
  toward conservative augmentation being the safer direction, which is
  what was already chosen in this project's implementation.

- **GenFlowchart (2024)** — combines Segment Anything Model (SAM) for
  visual segmentation with an LLM to assemble a symbolic representation.
  Another VLM-adjacent hybrid pipeline rather than a single trained
  end-to-end model.

- **BPMN structured extraction (Deka 2025)** — same lead author as
  Flowchart2Mermaid; VLM pipeline for extracting structured JSON from
  BPMN diagram images, enriched with OCR-based label recovery. Same
  "prompt a frontier VLM" family as Flowchart2Mermaid.

- **UML-specific work**: Tahuti (Hammond & Davis, sketch recognition via
  geometric properties, pre-deep-learning), "From Image to UML: First
  Results of Image-Based UML Diagram Generation using LLMs"
  ([arXiv:2404.11376](https://arxiv.org/abs/2404.11376), 2024, VLM
  prompting again), "Assessing GPT-4-Vision's Capabilities in UML-Based
  Code Generation" ([arXiv:2404.14370](https://arxiv.org/abs/2404.14370),
  2024). A 2025 Nature Scientific Reports survey ("Enhancing hand-drawn
  diagram recognition through the integration of machine learning and
  deep learning techniques") covers the broader hand-drawn-diagram-
  recognition space and its classical (pre-VLM) methods.

**Pattern across this whole section**: essentially every *recent*
(2024-2025) system in this space is "prompt a frontier VLM," not "train a
dedicated small model." The pre-VLM literature (GRCNN, Arrow R-CNN,
DrawnNet, Flowmind2Digital) trains dedicated models but targets *object
detection + structure assembly*, not direct end-to-end markup-text
generation. This project's approach — a small, purpose-trained,
end-to-end sequence-generation model targeting Mermaid specifically —
doesn't have a close match in either camp.

## 3. This project's actual architectural lineage: image → markup/code generation

The closest real ancestry for the TrOCR-encoder + small-transformer-
decoder design in `model.py` isn't the flowchart-recognition literature
above at all — it's the **image-to-markup generation** line of work,
which the flowchart papers rarely cite:

- **Im2Latex / "Image-to-Markup Generation with Coarse-to-Fine Attention"
  (Deng, Kanervisto, Ling, Rush,
  [arXiv:1609.04938](https://arxiv.org/abs/1609.04938), 2016)** — the
  foundational paper: CNN encoder + attention-based RNN decoder,
  trained end-to-end to convert an image of a math formula directly into
  LaTeX markup text, no OCR or object-detection step. This is the direct
  conceptual ancestor of this project's architecture, just swapping the
  target markup (LaTeX → Mermaid) and the modern building blocks
  (CNN+RNN+attention → TrOCR ViT encoder + transformer decoder).
  Established that attention-based *direct* markup generation
  outperforms classical OCR pipelines for this kind of non-standard-OCR
  task — a data point in favor of this project's own choice not to
  route through a separate OCR/object-detection stage.

- **"Teaching Machines to Code" (Singh, [arXiv:1802.05415]
  (https://arxiv.org/abs/1802.05415), 2018)** — same im2latex lineage,
  improved attention mechanism, BLEU 89% on formulas over 150 tokens
  long. Relevant data point on how far a purely end-to-end
  image-to-markup transducer can be pushed with enough attention-model
  refinement, without needing object detection or an OCR backbone at all.

- **Pix2Struct ([arXiv:2210.03347](https://arxiv.org/abs/2210.03347),
  2022)** — pretrains a ViT-encoder + transformer-decoder directly on
  massive screenshot→HTML pairs (variable-resolution patches, no
  distortion of aspect ratio), then fine-tunes for various
  visually-situated-language tasks. Architecturally the closest published
  system to this project's own encoder-decoder shape, though pretrained
  on web screenshots rather than an OCR-specific corpus like TrOCR.

- **MatCha ([arXiv:2212.09662](https://arxiv.org/abs/2212.09662), 2022)**
  — continues Pix2Struct pretraining with a "chart derendering" objective
  that explicitly **decodes a chart image into Python code or a data
  table**. This is the closest published framing to "decode a diagram
  image into structured/executable text" outside of Mermaid specifically,
  and supports the general premise that a compact ViT-encoder +
  transformer-decoder can learn this class of task without needing a
  giant general-purpose VLM.

- **DePlot ([arXiv via ACL 2023](https://arxiv.org/abs/2212.10505))** —
  built on Pix2Struct/MatCha, plot-image → data-table translation,
  designed so an LLM downstream can reason over the extracted table
  rather than the raw pixels. Same "small specialized derendering model
  as a front-end to a separate reasoning step" pattern this project
  doesn't need (Mermaid code is already the final target, not an
  intermediate representation for something else to consume).

## 4. On the render-feedback / RL fine-tuning idea discussed in conversation

Directly supports the "Future work: render-based (execution) reward"
section already in `IDEA.md` — this turns out to be an active, named
research direction, not a from-scratch idea:

- **RLRF — Reinforcement Learning from Rendering Feedback
  ([arXiv:2505.20793](https://arxiv.org/abs/2505.20793), NeurIPS 2025)**
  — nearly exactly the mechanism discussed in conversation, applied to
  SVG instead of Mermaid: a VLM generates multiple SVG "roll-outs" per
  input image, each is *rendered* (non-differentiable, exactly the same
  obstacle discussed for `mermaidx.render()`) and compared against the
  ground-truth image to compute a reward, which is then used for RL
  fine-tuning on top of a supervised-pretrained model. Directly confirms
  two things raised in conversation: (1) rendering-as-reward for
  non-differentiable markup generation is a real, working technique, and
  (2) **RLRF deliberately does NOT use raw pixel L2 alone** — its reward
  combines L2 distance, *semantic similarity* (DreamSim or CLIP
  embeddings), and a code-efficiency term. That three-part design is
  effectively the literature's own answer to the "raw pixel comparison
  can be a noisy reward for near-misses because layout is sensitive to
  small text changes" concern raised in this project's conversation —
  independent confirmation that a pure-pixel reward is understood as
  too fragile on its own in this exact family of technique, without
  contradicting the fact that mermaidx.render() is itself perfectly
  deterministic.

- **RefineSVG ([arXiv:2607.27699](https://arxiv.org/abs/2607.27699),
  2026)** — cites and builds on RLRF; adds a three-stage pipeline
  (open-loop SFT → rejection-sampling-based cold-start → agentic RL via
  GRPO with multi-dimensional rewards for structure/fidelity/efficiency),
  plus a closed-loop visual-diff-guided correction step at inference
  time. Explicitly names **"geometric drift"** as one of three failure
  modes of open-loop (non-RL) generation — independent terminology for
  essentially the same layout-cascade effect discussed re: Mermaid's
  auto-layout sensitivity to small label-text edits.

**Takeaway for this project's future-work section**: the SFT-then-RL
staging already proposed in `IDEA.md` (supervised cross-entropy first,
render-based RL fine-tuning only after convergence) matches how RLRF and
RefineSVG are actually structured in the literature, and their
multi-component reward (not raw pixel diff alone) is a concrete, tested
answer to exactly the objection about reward-landscape smoothness that
was raised and discussed for this project. If that future-work phase is
ever picked up, RLRF's reward formulation (L2 + semantic similarity +
efficiency, not L2 alone) is the most directly transferable starting
point found in the literature.
