"""Full independent-judge matrix: every (model, method) judged on single / unseen / triple
by three LLM judges of distinct families. Reports judge joint-success (fraction of samples
where ALL target attributes are rated >=3/5, averaged over judges). Mirrors the classifier
joint-success columns so the two can sit side by side in the main table."""
import _bootstrap  # noqa: F401
import json, re
from concurrent.futures import ThreadPoolExecutor
from gap_control.env import get_client

JUDGES = ["glm-5.2", "kimi-k2.6", "deepseek-v4-flash"]
ATTRS = {"single": ["positive sentiment"],
         "unseen": ["positive sentiment", "formal style"],
         "triple": ["positive sentiment", "formal style", "joyful emotion"]}
LIMIT = 100

# (model, method, {setting: (file, method_filter)})
CELLS = [
 ("Falcon","base",     {"single":("rb_base",None),"unseen":("rb_base",None),"triple":("rb_base",None)}),
 ("Falcon","prompting",{"single":("rb_pp_single","prompting"),"unseen":("rb_pp_ho2","prompting"),"triple":("rb_pp_tri","prompting")}),
 ("Falcon","PREADD",   {"single":("rb_pp_single","preadd"),"unseen":("rb_pp_ho2","preadd"),"triple":("rb_pp_tri","preadd")}),
 ("Falcon","LM-Steer", {"single":("con_lm_sentiment_positive",None),"unseen":("rb_lm_ho2_s6",None),"triple":("rb_lm_tri_s6",None)}),
 ("Falcon","GAP",      {"single":("con_gap_sentiment_positive",None),"unseen":("rb_gap2_ho2",None),"triple":("rb_gap2_tri",None)}),
 ("Falcon","FUDGE",    {"single":("fudge2_single",None),"unseen":("fudge2_ho2",None),"triple":("fudge2_tri",None)}),
 ("Falcon","BoN",      {"single":("rev_bon_single",None),"unseen":("rev_bon_ho2",None),"triple":("rev_bon_tri",None)}),
 ("Falcon","CD",       {"single":("cdmc_single",None),"unseen":("cdmc_ho2",None),"triple":("cdmc_tri",None)}),
 ("SmolLM2","base",     {"single":("sec_smol_base",None),"unseen":("sec_smol_base",None),"triple":("sec_smol_base",None)}),
 ("SmolLM2","prompting",{"single":("sec_smol_pp_single","prompting"),"unseen":("sec_smol_pp_ho2","prompting"),"triple":("sec_smol_pp_tri","prompting")}),
 ("SmolLM2","PREADD",   {"single":("sec_smol_pp_single","preadd"),"unseen":("sec_smol_pp_ho2","preadd"),"triple":("sec_smol_pp_tri","preadd")}),
 ("SmolLM2","LM-Steer", {"single":("sec_smol_lm_single",None),"unseen":("sec_smol_lm_ho2",None),"triple":("sec_smol_lm_tri",None)}),
 ("SmolLM2","GAP",      {"single":("sec_smol_gap_single",None),"unseen":("sec_smol_gap_ho2",None),"triple":("sec_smol_gap_tri",None)}),
]


def load(fname, mfilter):
    seen, rows = set(), []
    for l in open(f"outputs/{fname}.jsonl"):
        d = json.loads(l)
        if mfilter and d.get("method") != mfilter:
            continue
        if d["prompt_id"] in seen:
            continue
        seen.add(d["prompt_id"]); rows.append((d["prompt"], d["text"]))
        if len(rows) >= LIMIT:
            break
    return rows


def judge_one(model, attrs, prompt, text):
    lines = "\n".join(f"{i+1}) {a}" for i, a in enumerate(attrs))
    keys = ",".join(f'"{i+1}":x' for i in range(len(attrs)))
    msg = (f"Prompt: {prompt!r}\nText: {text!r}\n\nRate the TEXT on each attribute, integer 1-5 "
           f"(5 = strongly exhibits it):\n{lines}\nReply ONLY as JSON: {{{keys}}}.")
    extra = {"thinking": {"type": "disabled"}} if model.startswith("glm") else {}
    try:
        r = get_client().chat.completions.create(model=model, temperature=0.0, max_tokens=60,
              extra_body=extra, messages=[{"role": "user", "content": msg}])
        v = {int(k): int(val) for k, val in re.findall(r'"(\d+)"\s*:\s*([1-5])', r.choices[0].message.content or "")}
        if len(v) == len(attrs):
            return all(v[i+1] >= 3 for i in range(len(attrs)))
    except Exception:
        pass
    return None


def cell_value(fname, mfilter, attrs):
    rows = load(fname, mfilter)
    if not rows:
        return None
    per_judge = []
    for j in JUDGES:
        with ThreadPoolExecutor(max_workers=8) as ex:
            v = [x for x in ex.map(lambda pt: judge_one(j, attrs, pt[0], pt[1]), rows) if x is not None]
        if v:
            per_judge.append(sum(v) / len(v))
    return sum(per_judge) / len(per_judge) if per_judge else None


print(f"{'model':8} {'method':10} {'Jud-single':>11} {'Jud-unseen':>11} {'Jud-triple':>11}")
print("-" * 56)
for model, method, settings in CELLS:
    out = {}
    for s in ("single", "unseen", "triple"):
        fname, mf = settings[s]
        out[s] = cell_value(fname, mf, ATTRS[s])
    fmt = lambda x: f"{x:.2f}" if x is not None else "--"
    print(f"{model:8} {method:10} {fmt(out['single']):>11} {fmt(out['unseen']):>11} {fmt(out['triple']):>11}", flush=True)
