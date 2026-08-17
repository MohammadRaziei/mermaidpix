# mermaidpix-vlm

A small, fast "mini Donut" for image → Mermaid.js: one shared model
reconstructs the `.mmd` code for **any** supported diagram type, backed by
a tiny classifier that first says which type it is (or that it isn't a
Mermaid diagram at all).

## Why this architecture (and not object detection, and not a full VLM)

We went through three iterations before this one:

1. **Hand-written classical CV** (contours + Hough lines + OCR, zero
   learning) -- broke on real mermaid-rendered diagrams (filled shapes,
   curved arrows) because every geometric assumption was wrong for that
   style.
2. **Faster R-CNN shape/arrow detector + rule-based graph assembly** --
   worked, but only for flowcharts; every other diagram type (sequence,
   class, ER, gantt, pie, mindmap) has a genuinely different visual grammar
   and would have needed its own detector, its own category taxonomy, and
   its own assembly rules. Expensive to build and expensive to maintain.
3. **This version**: a small encoder-decoder that reads the image and
   writes the Mermaid code directly, the same *paradigm* full VLMs use
   (and the same paradigm behind **Donut** [Kim et al., ECCV 2022] and
   **Pix2Struct** [Lee et al., 2023]) but scaled down for a much narrower
   domain than general documents/screenshots.

Donut-base is 143M params, Pix2Struct-base is 282M. Both are far bigger
than the ~130MB (1/10th of a 1.3GB 4-bit model) target discussed. But both
were built for open-domain document/screenshot understanding. Our domain
(Mermaid syntax, a handful of diagram grammars) is dramatically narrower,
so we can shrink both halves of the architecture:

- **Encoder**: **updated** -- originally ViT-Tiny (ImageNet-pretrained,
  ~5.5M params); now swapped for TrOCR's BEiT-Base encoder (~86M params,
  pretrained specifically on reading text out of images, which matters
  since Mermaid diagrams are full of labels). Bottom 4 of 12 blocks frozen,
  top 8 fine-tuned. **See `IDEA.md` for the full rationale, size analysis,
  and exact frozen/trainable layer breakdown** -- this section only has the
  summary.
- **Decoder**: a small transformer decoder trained from scratch with a
  **custom BPE vocabulary trained on our own generated Mermaid corpus**
  (~4000 tokens) instead of reusing BART's full 50k multilingual
  vocabulary. That vocabulary size is most of what makes Donut's decoder
  heavy, and this domain doesn't need it.

Total: **~105M params** (see exact breakdown via `python model.py`,
which prints per-component counts without needing any data or a full
training run) -- roughly 400MB at fp32, 200MB at fp16. This grew from the
original ~41M/~150MB estimate when the encoder was swapped from
ImageNet ViT-Tiny to OCR-pretrained BEiT-Base -- a deliberate trade for
accuracy over size (see `IDEA.md`), per the priority stated when this
change was made. Still dramatically smaller than Donut (143M) or
Pix2Struct (282M).

**Research validation for going this small**: PP-OCRv5 (Baidu, 2025) is a
**5M-parameter** OCR model that matches or beats billion-parameter VLMs on
OCR benchmarks -- not through a bigger architecture, but through a
data-centric approach (more, cleaner, more diverse training data). We're
in an even better position than PP-OCRv5 here: since we generate the
training data ourselves (via real `mermaidx` renders), we can produce
unlimited, perfectly-labeled examples -- no annotation cost at all.

We also looked at DeepSeek-OCR before landing here: it's a strong general
OCR model, but at 3B total params / 570M active (MoE decoder) it's built
for a different problem (compressing arbitrary long documents into vision
tokens), was never evaluated on structured diagram reconstruction
specifically, and is still much heavier even 4-bit quantized than what's
needed for this narrow a task. Shrinking a general 3B model down 10x would
likely cost more effort and accuracy than training a purpose-built small
model on unlimited synthetic data from scratch.

## Architecture

```
image (PNG/JPG)
      │
      ▼
┌─────────────────────────────────────────────┐
│ Router: MobileNetV3-Small                    │  ~2.5M params, ImageNet-pretrained
│ 30 classes: not_diagram(0), plus 29 Mermaid   │
│ diagram types (flowchart, sequence, class,    │
│ state, er, gantt, pie, mindmap, swimlanes,    │
│ journey, quadrant, requirement, gitgraph, c4, │
│ timeline, sankey, xychart, block, packet,     │
│ kanban, architecture, radar, eventmodeling,   │
│ treemap, venn, ishikawa, wardley, cynefin,    │
│ treeview -- every type Mermaid documents      │
│ except ZenUML, an external plugin not on by   │
│ default; see below)                           │
└─────────────────────────────────────────────┘
      │
      ├── not_diagram ──► stop, report "not a mermaid diagram"
      │
      └── (any of the 29 diagram types) ──┐
                                ▼
                ┌───────────────────────────────────────┐
                │ Reconstructor (model.py)                │
                │  Encoder: TrOCR BEiT-Base (pretrained, ~86M)│
                │  Decoder: 4-layer transformer (scratch,  │
                │           custom ~4000-token vocabulary) │
                │  → autoregressively emits .mmd code       │
                └───────────────────────────────────────┘
```

Both models load once and stay resident -- there's no per-diagram-type
model swap, which is exactly the "loading cost" concern that motivated
moving away from the per-type-detector design.

## All 29 diagram types, and the one that's missing

Mermaid's own docs (`https://mermaid.ai/open-source/config/schema-docs/config.html`)
list every diagram type as a top-level config key. `common/diagram_generators.py`
has a generator for every one of them except **ZenUML**: ZenUML is shipped
as a separate package (`@mermaid-js/mermaid-zenuml`) that must be
explicitly registered via `mermaid.registerExternalDiagrams(...)` -- it's
not enabled by default in the core library, `mermaid-cli`, or `mermaidx`, so
generating training data for it isn't possible without extra setup on your
end. If you register the plugin yourself, adding a `build_zenuml()`
generator follows the exact same pattern as any other type here.

Several of the newer types (radar, eventmodeling, treemap, venn, ishikawa,
wardley, cynefin, treeview) are marked "beta"/experimental by Mermaid
itself (the 🔥 icon in their docs) and their syntax may still evolve --
worth knowing if a future Mermaid version breaks one of these generators.

## Theme and "look" randomization

Mermaid also has 11 themes (`default`, `base`, `dark`, `forest`, `neutral`,
`neo`, `neo-dark`, `redux`, `redux-dark`, `redux-color`, `redux-dark-color`)
and 3 "looks" (`classic`, `handDrawn`, `neo`) -- these change the visual
style, not the diagram content, so they are **not** router classes. Instead,
`generate_dataset.py` picks a random theme+look for every rendered image
(via Mermaid frontmatter, `random_theme_and_look` / `wrap_with_frontmatter`
in `common/diagram_generators.py`) while keeping the *target* text for the
reconstructor as the clean, unwrapped source -- so the model learns "any
visual style of a flowchart still becomes this flowchart code," instead of
overfitting to the default theme's look.

## What's tested vs. what isn't (read this before a long run)

- ✅ **Tested in my sandbox**: every one of the 29 diagram-source generators
  in `common/diagram_generators.py`, stress-tested at 15 random trials each
  (435 total) plus 10 negative-image trials, all rendering successfully
  through a real Mermaid engine -- **including** with random theme/look
  frontmatter applied (the full 435-trial matrix was re-run a second time
  with randomized theme+look wrapping and passed with zero failures).
- ✅ **Tested**: the tokenizer training pipeline (`train_tokenizer.py`) end
  to end on a real generated corpus -- round-trip encode/decode matched
  exactly.
- ✅ **Tested end-to-end**: `generate_dataset.py` itself, for real, in my
  sandbox -- this became possible once rendering moved from `mmdc`
  (mermaid-cli, needs a Chromium download that's blocked by my sandbox's
  network restrictions) to `mermaidx` (pure Python, no browser). Ran the
  full router + reconstructor generation across all 29 diagram types twice,
  with different random seeds, with theme/look randomization active: zero
  render failures, valid PNGs of sane size, and a correctly-formed
  `manifest.jsonl`. This run **caught and fixed a real bug**: the quadrant
  chart generator could produce a coordinate that rounds to display as
  `1.00` (e.g. `0.997` formatted with `:.2f`), which crashes that diagram
  type's parser with a lexical error -- confirmed by isolating it down to
  the exact triggering value, fixed by keeping generated coordinates at
  least 0.02 away from both 0 and 1.
- ⚠️ **Not tested**: `model.py`, `train_router.py`, `infer.py`, and the
  torch-dependent parts of `train_reconstructor.py` (the actual training
  loop / tensor pipeline). No GPU, no disk space for a `torch` install, and
  no network access to download the ~330MB `microsoft/trocr-base-stage1`
  checkpoint in my sandbox. The code uses standard, stable APIs
  (`transformers.VisionEncoderDecoderModel`, `torchvision.models`,
  `nn.TransformerDecoder`) and I'm confident in the logic, but I have not
  watched it actually train. Run `python model.py` first (random weights,
  no download, no GPU needed) to sanity-check the architecture wires
  together, then `python model.py --download` once you have network access
  to also confirm the real TrOCR checkpoint loads, before committing to a
  long training run.
- ✅ **Tested (torch-independent parts only)**: the on-the-fly data
  pipeline in `train_reconstructor.py` -- `SampleQueue` / `_producer_loop`,
  i.e. everything up to but not including the torchvision `transform`
  call. Verified with real multiprocessing in my sandbox: 2 producer
  processes filling a bounded `multiprocessing.Queue` up to its
  backpressure cap (confirmed producers correctly BLOCK in `put()` once
  full, and correctly resume once the consumer drains a few items), a
  consumer reading valid (PNG, metadata) pairs, the optional `--debug-dump`
  disk mirror producing valid same-named PNG+JSON pairs bounded to the
  expected count, and a clean shutdown (both processes joined without
  needing `terminate()`). Also caught and fixed two real bugs this way:
  (1) with `mermaidx` imported at *module* top-level, concurrent producer
  processes appeared to hang for 30+ seconds -- turned out to be
  single-core CPU contention on simultaneous ~9s QuickJS warmups (the
  sandbox has exactly 1 vCPU), not a bug, but `import mermaidx` was moved
  local to the producer function anyway as a safer default; (2) a real
  training run's `train.log` showed epoch time climbing ~3.4x over ~19
  epochs before suddenly dropping -- consistent with Windows Defender
  reacting to the original disk-spool design's constant small-file churn,
  which is why the queue itself moved to RAM (see "On-the-fly training
  data" below) and disk became an opt-in debug-only mirror.

## Run these in order

```bash
make install       # pure pip install, no Node/npm/Chromium needed
make smoke-test     # confirms mermaidx renders correctly (quick sanity check)
make data           # generates data/router/, reconstructor VAL images, tokenizer corpus
make tokenizer      # trains the ~6000-token BPE vocabulary on the generated corpus
make train-router   # diagram-type classifier
make train-reconstructor  # the image -> mermaid model (on-the-fly train data, default)
make package        # zips results/ into results.zip

# or just:
make all
```

Reconstructor **train** data is generated on-the-fly during training (see
"On-the-fly training data" below) -- `make data` only renders its fixed
**val** set plus a larger text-only corpus for the tokenizer, so there's no
separate "render N training images to disk" step anymore. To compare
against the old fixed-image approach directly: `make data-fixed && make
train-reconstructor-fixed`.

The real training queue lives in RAM, so there's nothing to inspect on
disk by default. Pass `--debug-dump` to `train_reconstructor.py` to also
mirror recent samples to disk (cleared every epoch), then in a separate
terminal:
```bash
python inspect_on_the_fly.py --debug-dir ./results/reconstructor/otf_debug --watch
```

Then send me `results.zip`. Each training script also prints an explicit
list of what to paste back in the terminal output, but the zip has
everything already: `results/router/{summary.json,history.json,router_model.pt}`
and `results/reconstructor/{summary.json,history.json,qualitative_samples.json,roundtrip/,reconstructor_model.pt}`.

`make package` excludes model `.pt` checkpoints by default now (bandwidth)
-- use `make package-with-checkpoints` to include them.

## On-the-fly training data

`train_reconstructor.py` defaults to generating its training data live
instead of reading pre-rendered images off disk (`--no-on-the-fly` reverts
to the old fixed-manifest behavior, for comparison). This matters because
the model's original hallucinated-label problem came from training on a
*finite* set of pre-rendered images -- no matter how large the label
vocabulary, a fixed set of images is something a model can partially
memorize instead of actually reading. On-the-fly generation makes that
shortcut unavailable: no two training samples are ever the same image
twice. The **val** split stays fixed (needed for val_loss to be comparable
across epochs/runs).

It's implemented as an in-RAM producer/consumer queue (`SampleQueue` /
`_producer_loop` in `train_reconstructor.py`, backed by
`multiprocessing.Queue`), not a plain `DataLoader(num_workers=N)`:
`--num-workers` producer OS processes each independently render random
diagrams via `mermaidx` and `put()` them straight onto the queue -- OS
pipes, no disk. `put()`/`get()` block/unblock automatically at
`--queue-depth-batches` (default 10 batches' worth of samples), which is
the backpressure, natively, no polling loop needed. This used to be a
disk-backed spool directory specifically so an independent process could
inspect the live queue -- moved to RAM after a real run's `train.log`
showed evidence of Windows Defender-driven slowdown from that design's
constant small-file churn (see "What's tested" above and IDEA.md's Round
2/3 analysis). The inspectability goal is preserved as an opt-in
`--debug-dump` flag instead: producers also mirror each sample to a small,
bounded, per-worker set of disk slots (PNG + a same-named `.json` with the
generation info -- target Mermaid source, theme, look, engine), cleared
once per epoch, purely for `inspect_on_the_fly.py` to look at.

## Round-trip test

At the end of every `train_reconstructor.py` run, `roundtrip_test.py` runs
automatically: a small, fixed, deterministic set (one example per diagram
type, same every run) gets rendered with mermaidx's own quickjs backend,
fed through the just-trained model, and the (ground_truth, prediction)
pairs get written to `results/reconstructor/roundtrip/` --
`roundtrip_report.txt` (human-eyeballable), `roundtrip_report.json`
(machine-readable), and `images/` (what the model actually saw). Re-run
standalone against any saved checkpoint with `make roundtrip` (or `python
roundtrip_test.py --model <path> --tokenizer <path>`), without retraining.

## Files

| File | Purpose |
|---|---|
| `IDEA.md` | design doc for the TrOCR-encoder architecture: rationale, diagram, size analysis, frozen/trainable layers |
| `common/diagram_generators.py` | random, validated Mermaid source generators for all 29 types + negatives + theme/look randomization |
| `generate_dataset.py` | renders the router dataset + reconstructor VAL images + tokenizer text corpus (Step 1) |
| `train_tokenizer.py` | trains the small BPE vocabulary (Step 2) |
| `model.py` | the TrOCR-encoder + small-decoder architecture; `python model.py` prints param counts (add `--download` to also verify the real checkpoint loads) |
| `train_router.py` | trains the diagram-type classifier (Step 3) |
| `train_reconstructor.py` | trains the image->code model (Step 4) -- on-the-fly training data by default, see above |
| `hf_offline_first.py` | shared helper: load a HuggingFace checkpoint from local cache only, falling back to a real download just once if it isn't cached yet -- avoids repeated network attempts on a flaky connection |
| `roundtrip_test.py` | mmd -> quickjs png -> model -> mmd sanity check, run automatically after training (see above), or standalone via `make roundtrip` |
| `inspect_on_the_fly.py` | separate, read-only tool to watch the on-the-fly training data (only useful with `--debug-dump`, see above) |
| `infer.py` | run the finished pipeline on a real image (Step 5, after training) |
| `package_results.py` | zips `results/` for you to send back |
| `Makefile` | orchestrates all of the above in order |
| `exclude_from_defender.bat` | Windows-only: adds a Defender exclusion for this folder, to test the antivirus-slowdown hypothesis above |

## Extending further

- **More diagram-type coverage**: add a new builder function to
  `common/diagram_generators.py` and it's automatically included in both
  datasets and the router's class list -- no other code changes needed.
  This is the main advantage of the seq2seq approach over per-type object
  detection: adding a diagram type is a data problem, not an architecture
  problem.
- **Real-world generalization**: everything here is trained on
  mermaidx's rendering style (which itself wraps mermaid.js, same as the
  official renderer). If your actual use case includes
  diagrams from other tools (draw.io, hand-drawn, screenshots from other
  apps), mix a manually-collected batch of those in before trusting the
  model on them -- the synthetic data covers Mermaid's visual style well,
  but nothing outside it.
- **Shrinking further**: once accuracy looks good, standard post-training
  quantization (int8/int4) should get the checkpoint well under the 15MB
  mark with minimal accuracy loss, since the model is already small and
  well within a normal quantization regime.
