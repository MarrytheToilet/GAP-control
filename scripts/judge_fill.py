"""Fill the remaining single-attribute Judge cells (independent-judge mean relevance,
mean of 3 LLM judges) for methods/models not yet judged, with method filtering."""
import _bootstrap  # noqa: F401
import json, re
from concurrent.futures import ThreadPoolExecutor
from gap_control.env import get_client

JUDGES = ["glm-5.2", "kimi-k2.6", "deepseek-v4-flash"]
ATTR = "positive sentiment"
LIMIT = 100
CELLS = [
    ("Falcon base",   "rb_base",            None),
    ("Falcon PREADD", "rb_pp_single",       "preadd"),
    ("Smol base",     "sec_smol_base",      None),
    ("Smol prompting","sec_smol_pp_single", "prompting"),
    ("Smol PREADD",   "sec_smol_pp_single", "preadd"),
    ("Smol LM-Steer", "sec_smol_lm_single", None),
    ("Smol GAP",      "sec_smol_gap_single",None),
]


def load(fname, mf):
    seen, rows = set(), []
    for l in open(f"outputs/{fname}.jsonl"):
        d = json.loads(l)
        if mf and d.get("method") != mf:
            continue
        if d["prompt_id"] in seen:
            continue
        seen.add(d["prompt_id"]); rows.append((d["prompt"], d["text"]))
        if len(rows) >= LIMIT:
            break
    return rows


def score_one(model, prompt, text):
    msg = (f"Prompt: {prompt!r}\nText: {text!r}\n\nRate the TEXT for the attribute "
           f'"{ATTR}" on an integer scale 1-5 (5 = clearly {ATTR}). '
           'Reply ONLY as JSON: {"relevance":x}.')
    extra = {"thinking": {"type": "disabled"}} if model.startswith("glm") else {}
    try:
        r = get_client().chat.completions.create(model=model, temperature=0.0, max_tokens=40,
              extra_body=extra, messages=[{"role": "user", "content": msg}])
        m = re.search(r'"relevance"\s*:\s*([1-5])', r.choices[0].message.content or "")
        if m:
            return (int(m.group(1)) - 1) / 4.0
    except Exception:
        pass
    return None


for label, fname, mf in CELLS:
    rows = load(fname, mf)
    per_judge = []
    for j in JUDGES:
        with ThreadPoolExecutor(max_workers=8) as ex:
            v = [x for x in ex.map(lambda pt: score_one(j, pt[0], pt[1]), rows) if x is not None]
        if v:
            per_judge.append(sum(v) / len(v))
    val = sum(per_judge) / len(per_judge) if per_judge else float("nan")
    print(f"{label:16s} judge-rel = {val:.2f}  (n={len(rows)})", flush=True)
