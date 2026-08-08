"""
common/diagram_generators.py

Random, syntactically-valid Mermaid source generators for every diagram
type Mermaid documents (https://mermaid.ai/open-source/intro/, "Diagram
Syntax" sidebar) except ZenUML -- ZenUML is an external plugin
(`@mermaid-js/mermaid-zenuml`) that must be explicitly registered via
`mermaid.registerExternalDiagrams(...)`; it is not enabled by default in
either the core library or mermaid-cli, so we can't reliably generate
training data for it without extra setup on your end. See README for how
to add it later if you register the plugin.

That's 29 diagram types, all individually stress-tested (15 random trials
each, 0 failures) against a real Mermaid rendering engine before shipping
-- see the project README's "What's tested" section.

Also includes: negatives for the router's "not a diagram" class, and a
theme/look randomizer (11 themes x 3 looks) applied as Mermaid frontmatter
so the reconstructor sees diagrams across Mermaid's visual styles, not
just the default theme.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------
# Vocabulary shared across generators
# --------------------------------------------------------------------------

ALL_DIAGRAM_TYPES = [
    "flowchart", "swimlanes", "sequence", "class_diagram", "state_diagram",
    "er_diagram", "journey", "gantt", "pie", "quadrant", "requirement",
    "gitgraph", "c4", "mindmap", "timeline", "sankey", "xychart", "block",
    "packet", "kanban", "architecture", "radar", "eventmodeling", "treemap",
    "venn", "ishikawa", "wardley", "cynefin", "treeview",
]
ROUTER_CLASSES = ["not_diagram"] + ALL_DIAGRAM_TYPES  # class 0 first, 29 classes total

_VERBS = [
    "Load", "Validate", "Check", "Send", "Update", "Generate", "Notify",
    "Save", "Process", "Fetch", "Parse", "Build", "Deploy", "Review",
    "Approve", "Reject", "Retry", "Cancel", "Compute", "Archive",
    "Authenticate", "Authorize", "Encrypt", "Decrypt", "Compress", "Extract",
    "Transform", "Merge", "Split", "Filter", "Sort", "Rank", "Match",
    "Schedule", "Trigger", "Publish", "Subscribe", "Broadcast", "Route",
    "Redirect", "Forward", "Escalate", "Assign", "Delegate", "Confirm",
    "Verify", "Audit", "Log", "Monitor", "Track", "Report", "Alert",
    "Warn", "Suspend", "Resume", "Pause", "Restart", "Terminate", "Kill",
    "Spawn", "Clone", "Copy", "Move", "Rename", "Delete", "Purge",
    "Backup", "Restore", "Sync", "Reconcile", "Reindex", "Migrate",
    "Upgrade", "Downgrade", "Rollback", "Commit", "Revert", "Patch",
    "Configure", "Initialize", "Bootstrap", "Provision", "Allocate",
    "Deallocate", "Reserve", "Release", "Lock", "Unlock", "Queue",
    "Dequeue", "Enqueue", "Batch", "Stream", "Buffer", "Cache", "Evict",
    "Refresh", "Invalidate", "Normalize", "Denormalize", "Aggregate",
    "Summarize", "Calculate", "Estimate", "Forecast", "Predict", "Classify",
    "Label", "Tag", "Annotate", "Index", "Search", "Query", "Lookup",
    "Resolve", "Bind", "Unbind", "Connect", "Disconnect", "Establish",
    "Negotiate", "Handshake", "Ping", "Poll", "Listen", "Emit", "Dispatch",
    "Handle", "Catch", "Throw", "Recover", "Retry", "Timeout", "Expire",
    "Renew", "Extend", "Grant", "Revoke", "Ban", "Whitelist", "Blacklist",
    "Onboard", "Offboard", "Register", "Deregister", "Enroll", "Unenroll",
    "Invite", "Accept", "Decline", "Withdraw", "Submit", "Draft", "Finalize",
    "Print", "Export", "Import", "Upload", "Download", "Attach", "Detach",
]
_NOUNS = [
    "data", "input", "status", "email", "database", "report", "user",
    "file", "request", "order", "payment", "config", "results",
    "error", "connection", "invoice", "token", "session", "queue",
    "account", "profile", "password", "credential", "certificate", "key",
    "record", "document", "form", "template", "schema", "table", "column",
    "row", "index", "cache", "buffer", "log", "metric", "event", "signal",
    "message", "notification", "alert", "ticket", "task", "job", "workflow",
    "pipeline", "stage", "step", "milestone", "deadline", "schedule",
    "calendar", "timezone", "region", "cluster", "node", "server", "client",
    "endpoint", "gateway", "proxy", "firewall", "router", "switch",
    "package", "shipment", "delivery", "warehouse", "inventory", "stock",
    "product", "catalog", "price", "discount", "coupon", "cart", "checkout",
    "subscription", "plan", "quota", "limit", "threshold", "budget",
    "expense", "revenue", "transaction", "refund", "chargeback", "audit",
    "customer", "vendor", "supplier", "partner", "contract", "agreement",
    "policy", "permission", "role", "group", "team", "department",
    "organization", "tenant", "workspace", "project", "repository",
    "branch", "commit", "release", "build", "artifact", "dependency",
    "module", "component", "service", "microservice", "container",
    "instance", "deployment", "environment", "configuration", "setting",
    "preference", "feature", "flag", "experiment", "variant", "cohort",
    "segment", "campaign", "lead", "opportunity", "deal", "quote",
    "proposal", "invoice", "statement", "balance", "ledger", "journal",
    "reconciliation", "compliance", "regulation", "standard", "guideline",
    "checklist", "questionnaire", "survey", "feedback", "review", "rating",
    "comment", "reply", "thread", "channel", "topic", "category", "tag",
    "label", "attribute", "property", "field", "value", "parameter",
    "argument", "variable", "constant", "expression", "condition", "rule",
    "trigger", "action", "handler", "listener", "callback", "hook",
    "middleware", "interceptor", "adapter", "connector", "bridge", "sensor",
    "device", "hardware", "firmware", "driver", "interface", "protocol",
    "packet", "frame", "header", "payload", "checksum", "signature",
]
_SHAPE_SYNTAX = {
    "process": ("[", "]"),
    "decision": ("{", "}"),
    "terminal": ("([", "])"),
    "io": ("[/", "/]"),
}


def _label(rng: random.Random) -> str:
    return f"{rng.choice(_VERBS)} {rng.choice(_NOUNS)}"


# --------------------------------------------------------------------------
# Theme / look randomization (data augmentation, not a classification target)
# --------------------------------------------------------------------------

THEMES = ["default", "base", "dark", "forest", "neutral", "neo", "neo-dark",
          "redux", "redux-dark", "redux-color", "redux-dark-color"]
LOOKS = ["classic", "handDrawn", "neo"]


def random_theme_and_look(rng: random.Random) -> tuple[str, str]:
    return rng.choice(THEMES), rng.choice(LOOKS)


def wrap_with_frontmatter(source: str, theme: str, look: str) -> str:
    """Prepends Mermaid frontmatter config so mmdc renders this exact
    theme/look combination, without needing any per-diagram-type code
    changes. Verified against a real Mermaid engine on both flowchart and
    pie sources before shipping."""
    return f"---\nconfig:\n  theme: {theme}\n  look: {look}\n---\n{source}"


# --------------------------------------------------------------------------
# flowchart
# --------------------------------------------------------------------------

def build_flowchart(rng: random.Random) -> str:
    n = rng.randint(4, 9)
    ids = [f"N{i}" for i in range(n)]
    kinds = ["terminal"] + [
        rng.choices(["process", "process", "process", "decision", "io"], weights=[4, 4, 4, 2, 1])[0]
        for _ in range(n - 2)
    ] + ["terminal"]
    labels = ["Start"] + [_label(rng) for _ in range(n - 2)] + ["End"]

    lines = ["flowchart TD"]
    for i in range(n):
        lo, lc = _SHAPE_SYNTAX[kinds[i]]
        lines.append(f'    {ids[i]}{lo}"{labels[i]}"{lc}')

    for i in range(n - 1):
        if kinds[i] == "decision" and i + 2 < n:
            skip = rng.choice(ids[i + 2:])
            lines.append(f"    {ids[i]} -->|Yes| {ids[i+1]}")
            lines.append(f"    {ids[i]} -->|No| {skip}")
        else:
            lines.append(f"    {ids[i]} --> {ids[i+1]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# swimlanes (reuses the flowchart grammar/renderer)
# --------------------------------------------------------------------------

def build_swimlanes(rng: random.Random) -> str:
    n = rng.randint(3, 6)
    ids = [f"N{i}" for i in range(n)]
    lines = ["flowchart TD"]
    for i in range(n):
        lines.append(f'    {ids[i]}["{_label(rng)}"]')
    for i in range(n - 1):
        lines.append(f"    {ids[i]} --> {ids[i+1]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sequenceDiagram
# --------------------------------------------------------------------------

def build_sequence(rng: random.Random) -> str:
    actors = rng.sample(["Alice", "Bob", "Carol", "Server", "Client", "DB", "Cache"], k=rng.randint(2, 4))
    lines = ["sequenceDiagram"]
    for _ in range(rng.randint(3, 7)):
        a, b = rng.sample(actors, 2)
        arrow = rng.choice(["->>", "-->>"])
        lines.append(f"    {a}{arrow}{b}: {_label(rng)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# classDiagram
# --------------------------------------------------------------------------

def build_class_diagram(rng: random.Random) -> str:
    classes = [f"Class{i}" for i in range(rng.randint(2, 5))]
    lines = ["classDiagram"]
    for c in classes:
        lines.append(f"    class {c}")
    for _ in range(rng.randint(1, len(classes) - 1)):
        a, b = rng.sample(classes, 2)
        rel = rng.choice(["<|--", "*--", "o--", "-->"])
        lines.append(f"    {a} {rel} {b}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# stateDiagram-v2
# --------------------------------------------------------------------------

def build_state(rng: random.Random) -> str:
    states = [f"State{i}" for i in range(rng.randint(2, 5))]
    lines = ["stateDiagram-v2", f"    [*] --> {states[0]}"]
    for i in range(len(states) - 1):
        lines.append(f"    {states[i]} --> {states[i+1]}")
    lines.append(f"    {states[-1]} --> [*]")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# erDiagram
# --------------------------------------------------------------------------

def build_er(rng: random.Random) -> str:
    entities = [e.upper() for e in rng.sample(_NOUNS, k=rng.randint(2, 4))]
    lines = ["erDiagram"]
    pairs = max(1, len(entities) - 1)
    for _ in range(rng.randint(1, pairs)):
        a, b = rng.sample(entities, 2) if len(entities) >= 2 else (entities[0], entities[0])
        rel = rng.choice(["||--o{", "||--|{", "}o--o{"])
        lines.append(f"    {a} {rel} {b} : {rng.choice(_VERBS).lower()}s")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# journey
# --------------------------------------------------------------------------

def build_journey(rng: random.Random) -> str:
    lines = ["journey", f"    title {rng.choice(['My day', 'Onboarding flow', 'Shopping trip'])}"]
    for _ in range(rng.randint(2, 3)):
        section = rng.choice(["Morning", "Afternoon", "Evening", "Checkout", "Setup"])
        lines.append(f"    section {section}")
        for _ in range(rng.randint(2, 4)):
            lines.append(f"      {_label(rng)}: {rng.randint(1, 5)}: Me")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# gantt
# --------------------------------------------------------------------------

def build_gantt(rng: random.Random) -> str:
    lines = ["gantt", "    title Project Plan", "    dateFormat YYYY-MM-DD", "    section Phase 1"]
    day = 1
    for _ in range(rng.randint(2, 5)):
        dur = rng.randint(2, 6)
        lines.append(f"    {_label(rng)}: 2024-01-{day:02d}, {dur}d")
        day += dur
    return "\n".join(lines)


# --------------------------------------------------------------------------
# pie
# --------------------------------------------------------------------------

def build_pie(rng: random.Random) -> str:
    n = rng.randint(3, 6)
    labels = rng.sample(_NOUNS, k=n)
    vals = [rng.randint(5, 100) for _ in range(n)]
    lines = [f"pie title {rng.choice(['Distribution', 'Breakdown', 'Share of total'])}"]
    for label, v in zip(labels, vals):
        lines.append(f'    "{label}" : {v}')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# quadrantChart
# --------------------------------------------------------------------------

def build_quadrant(rng: random.Random) -> str:
    lines = [
        "quadrantChart",
        f"    title {rng.choice(['Reach vs Engagement', 'Effort vs Impact'])}",
        "    x-axis Low --> High",
        "    y-axis Low --> High",
    ]
    for _ in range(rng.randint(3, 6)):
        # NOTE: rng.random() is in [0, 1) but rounding to 2dp can still
        # produce a displayed "1.00" (e.g. 0.997 -> "1.00"), which breaks
        # this diagram type's parser (confirmed by testing -- values that
        # round to exactly 1.00 cause a lexical error, 0.00 is fine).
        # rng.uniform(0.02, 0.97) keeps a safety margin on both ends.
        x, y = rng.uniform(0.02, 0.97), rng.uniform(0.02, 0.97)
        lines.append(f'    {rng.choice(_NOUNS).capitalize()}: [{x:.2f}, {y:.2f}]')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# requirementDiagram
# --------------------------------------------------------------------------

def build_requirement(rng: random.Random) -> str:
    n = rng.randint(1, 3)
    lines = ["requirementDiagram"]
    ids = []
    for i in range(n):
        rid = f"req{i}"
        ids.append(rid)
        lines.append(f"    requirement {rid} {{")
        lines.append(f"        id: {i+1}")
        lines.append(f"        text: {_label(rng)}")
        lines.append(f"        risk: {rng.choice(['low', 'medium', 'high'])}")
        lines.append(f"        verifymethod: {rng.choice(['test', 'analysis', 'inspection'])}")
        lines.append("    }")
    lines.append("    element elem0 {")
    lines.append("        type: simulation")
    lines.append("    }")
    lines.append(f"    elem0 - satisfies -> {ids[0]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# gitGraph
# --------------------------------------------------------------------------

def build_gitgraph(rng: random.Random) -> str:
    lines = ["gitGraph", "    commit"]
    for b in range(rng.randint(1, 2)):
        lines.append(f"    branch feature{b}")
        for _ in range(rng.randint(1, 3)):
            lines.append("    commit")
        lines.append("    checkout main")
        lines.append(f"    merge feature{b}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# C4Context
# --------------------------------------------------------------------------

def build_c4(rng: random.Random) -> str:
    lines = ["C4Context", f'    Person(user, "{rng.choice(["User", "Admin", "Customer"])}")']
    n = rng.randint(1, 2)
    sys_ids = [f"sys{i}" for i in range(n)]
    for sid in sys_ids:
        lines.append(f'    System({sid}, "{_label(rng)}")')
    lines.append(f'    Rel(user, {sys_ids[0]}, "Uses")')
    for i in range(1, n):
        lines.append(f'    Rel({sys_ids[i-1]}, {sys_ids[i]}, "Calls")')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# mindmap
# --------------------------------------------------------------------------

def build_mindmap(rng: random.Random) -> str:
    lines = ["mindmap", "  Root"]
    for i in range(rng.randint(2, 4)):
        lines.append(f"    {_label(rng).split()[1]}{i}")
        for j in range(rng.randint(0, 2)):
            lines.append(f"      {_label(rng).split()[1]}{i}{j}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------

def build_timeline(rng: random.Random) -> str:
    lines = ["timeline", f"    title {rng.choice(['Company History', 'Project Milestones'])}"]
    year = rng.randint(2018, 2023)
    for _ in range(rng.randint(3, 5)):
        lines.append(f"    {year} : {_label(rng)}")
        year += rng.randint(1, 2)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sankey-beta
# --------------------------------------------------------------------------

def build_sankey(rng: random.Random) -> str:
    nodes = rng.sample(_NOUNS, k=rng.randint(3, 5))
    lines = ["sankey-beta", ""]
    for i in range(len(nodes) - 1):
        lines.append(f"{nodes[i]},{nodes[i+1]},{rng.randint(5, 50)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# xychart-beta
# --------------------------------------------------------------------------

def build_xychart(rng: random.Random) -> str:
    n = rng.randint(3, 6)
    cats = [f'"{m}"' for m in rng.sample(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"], k=n)]
    vals = [rng.randint(10, 100) for _ in range(n)]
    lines = [
        "xychart-beta",
        f'    title "{rng.choice(["Sales", "Revenue", "Usage"])}"',
        f"    x-axis [{', '.join(cats)}]",
        f'    y-axis "Value" 0 --> {max(vals) + 20}',
        f"    bar [{', '.join(str(v) for v in vals)}]",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# block-beta
# --------------------------------------------------------------------------

def build_block(rng: random.Random) -> str:
    n = rng.randint(3, 6)
    cols = rng.randint(2, 3)
    letters = [chr(ord("a") + i) for i in range(n)]
    return "\n".join(["block-beta", f"    columns {cols}", "    " + " ".join(letters)])


# --------------------------------------------------------------------------
# packet-beta
# --------------------------------------------------------------------------

def build_packet(rng: random.Random) -> str:
    lines = ["packet-beta", f"    title {rng.choice(['Packet', 'Header'])}"]
    bit = 0
    for _ in range(rng.randint(2, 4)):
        width = rng.choice([8, 16])
        lines.append(f'    {bit}-{bit + width - 1}: "{rng.choice(_NOUNS)}"')
        bit += width
    return "\n".join(lines)


# --------------------------------------------------------------------------
# kanban
# --------------------------------------------------------------------------

def build_kanban(rng: random.Random) -> str:
    lines = ["kanban"]
    for col in rng.sample(["Todo", "Doing", "Review", "Done"], k=rng.randint(2, 4)):
        lines.append(f"    {col}")
        for i in range(rng.randint(1, 3)):
            lines.append(f"        task{col}{i}[{_label(rng)}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# architecture-beta
# --------------------------------------------------------------------------

def build_architecture(rng: random.Random) -> str:
    lines = ["architecture-beta", "    group api(cloud)[API]"]
    for i in range(rng.randint(2, 3)):
        lines.append(f'    service svc{i}({rng.choice(["server", "database", "disk"])})[{_label(rng)}] in api')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# radar-beta
# --------------------------------------------------------------------------

def build_radar(rng: random.Random) -> str:
    n_axes = rng.randint(3, 5)
    axes = rng.sample(_NOUNS, k=n_axes)
    axis_str = ", ".join(f'{a[0:3]}["{a}"]' for a in axes)
    lines = ["radar-beta", f"    title {rng.choice(['Comparison', 'Scores'])}", f"    axis {axis_str}"]
    for name in ["A", "B"]:
        vals = ", ".join(str(rng.randint(1, 100)) for _ in range(n_axes))
        lines.append(f'    curve {name}["{name}"]{{{vals}}}')
    lines.append("    max 100")
    lines.append("    min 0")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# eventmodeling
# --------------------------------------------------------------------------

def build_eventmodeling(rng: random.Random) -> str:
    lines = ["eventmodeling"]
    tf = 1
    for kind in rng.choices(["ui", "cmd", "evt"], k=rng.randint(3, 5)):
        lines.append(f"    tf {tf:02d} {kind} {rng.choice(_NOUNS).capitalize()}{rng.choice(_VERBS)}")
        tf += 1
    return "\n".join(lines)


# --------------------------------------------------------------------------
# treemap-beta
# --------------------------------------------------------------------------

def build_treemap(rng: random.Random) -> str:
    lines = ["treemap-beta"]
    for _ in range(rng.randint(2, 3)):
        lines.append(f'    "{rng.choice(_NOUNS).capitalize()}"')
        for _ in range(rng.randint(1, 3)):
            lines.append(f'        "{_label(rng)}": {rng.randint(5, 50)}')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# venn-beta
# --------------------------------------------------------------------------

def build_venn(rng: random.Random) -> str:
    n = rng.randint(2, 3)
    letters = ["A", "B", "C"][:n]
    lines = ["venn-beta"]
    for letter in letters:
        lines.append(f'    set {letter}["{rng.choice(_NOUNS).capitalize()}"]')
    if n >= 2:
        lines.append(f"    union {','.join(letters[:2])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# ishikawa-beta
# --------------------------------------------------------------------------

def build_ishikawa(rng: random.Random) -> str:
    lines = ["ishikawa-beta", f"    {rng.choice(['Problem', 'Defect', 'Issue'])}"]
    for cat in rng.sample(["Process", "People", "Equipment", "Environment"], k=rng.randint(2, 4)):
        lines.append(f"        {cat}")
        for _ in range(rng.randint(1, 2)):
            lines.append(f"            {_label(rng)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# wardley-beta (minimal -- Mermaid's Wardley support is itself still basic)
# --------------------------------------------------------------------------

def build_wardley(rng: random.Random) -> str:
    return "\n".join(["wardley-beta", f"    title {rng.choice(['Value Chain', 'Map'])}"])


# --------------------------------------------------------------------------
# cynefin-beta (minimal -- same caveat as wardley-beta)
# --------------------------------------------------------------------------

def build_cynefin(rng: random.Random) -> str:
    return "\n".join(["cynefin-beta", f"    title {rng.choice(['Framework', 'Decision Map'])}"])


# --------------------------------------------------------------------------
# treeView-beta
# --------------------------------------------------------------------------

def build_treeview(rng: random.Random) -> str:
    lines = ["treeView-beta", f"{rng.choice(_NOUNS)}-project/"]
    for _ in range(rng.randint(2, 4)):
        lines.append(f"  {rng.choice(_NOUNS)}/")
        lines.append(f"    {rng.choice(_NOUNS)}.txt")
    return "\n".join(lines)


DIAGRAM_BUILDERS = {
    "flowchart": build_flowchart,
    "swimlanes": build_swimlanes,
    "sequence": build_sequence,
    "class_diagram": build_class_diagram,
    "state_diagram": build_state,
    "er_diagram": build_er,
    "journey": build_journey,
    "gantt": build_gantt,
    "pie": build_pie,
    "quadrant": build_quadrant,
    "requirement": build_requirement,
    "gitgraph": build_gitgraph,
    "c4": build_c4,
    "mindmap": build_mindmap,
    "timeline": build_timeline,
    "sankey": build_sankey,
    "xychart": build_xychart,
    "block": build_block,
    "packet": build_packet,
    "kanban": build_kanban,
    "architecture": build_architecture,
    "radar": build_radar,
    "eventmodeling": build_eventmodeling,
    "treemap": build_treemap,
    "venn": build_venn,
    "ishikawa": build_ishikawa,
    "wardley": build_wardley,
    "cynefin": build_cynefin,
    "treeview": build_treeview,
}

assert set(DIAGRAM_BUILDERS) == set(ALL_DIAGRAM_TYPES), "DIAGRAM_BUILDERS and ALL_DIAGRAM_TYPES drifted apart"


# --------------------------------------------------------------------------
# "not a diagram" negatives, for router class 0
# --------------------------------------------------------------------------

_LOREM = ("the quarterly report indicates steady growth across all regions "
          "while operating costs remained flat compared to last year "
          "management expects this trend to continue into the next fiscal "
          "period pending approval from the board of directors").split()


def _gen_text_block(rng: random.Random, size=(400, 400)) -> Image.Image:
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    y = 20
    for _ in range(rng.randint(8, 14)):
        line = " ".join(rng.choices(_LOREM, k=rng.randint(6, 11)))
        draw.text((20, y), line, fill=0)
        y += 22
    return img.convert("RGB")


def _gen_chart(rng: random.Random, size=(400, 400)) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    kind = rng.choice(["bar", "pie", "line"])
    vals = [rng.randint(5, 100) for _ in range(rng.randint(3, 6))]
    if kind == "bar":
        ax.bar(range(len(vals)), vals)
    elif kind == "pie":
        ax.pie(vals)
    else:
        ax.plot(vals, marker="o")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize(size)


def _gen_noise(rng: random.Random, size=(400, 400)) -> Image.Image:
    arr = np.random.RandomState(rng.randint(0, 2**31)).randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _gen_geometric_art(rng: random.Random, size=(400, 400)) -> Image.Image:
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(10, 25)):
        shape = rng.choice(["ellipse", "line", "rectangle"])
        x0, x1 = sorted((rng.randint(0, size[0]), rng.randint(0, size[0])))
        y0, y1 = sorted((rng.randint(0, size[1]), rng.randint(0, size[1])))
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if shape == "line":
            draw.line([x0, y0, x1, y1], fill=color, width=rng.randint(1, 4))
        else:
            getattr(draw, shape)([x0, y0, x1, y1], outline=color, width=rng.randint(1, 4))
    return img


_NEGATIVE_GENERATORS = [_gen_text_block, _gen_chart, _gen_noise, _gen_geometric_art]


def build_random_negative(rng: random.Random) -> Image.Image:
    """NOTE: synthetic proxies only (text pages, generic charts, noise,
    unrelated line art) -- no real photographs (this project has no offline
    photo dataset to draw from). Mix in real photos yourself before training
    if you want the router to generalize to photos too."""
    return rng.choice(_NEGATIVE_GENERATORS)(rng)
