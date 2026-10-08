"""Different STARTING parameter sets of the technical indicators (owner, 2026-10-08: "add different initial parameter
builds of the technical indicators to the queue"). Same tiny layout as ind1 (1-day block, excerpt, 6 climb slices x 2
seeds, all 9 heads). All 14 families x 3 instances; only the starting periods change; every period is clipped to the
learnable range [2, 60] (MOMENTUM_CLIP_MIN, LOOKBACK). Fixed before any run:
  textbook      today's defaults (= ind1_all, re-run here so the comparison shares a queue)
  short         every period x0.5
  long          every period x2
  wide          per family, the 3 instances spread to the short/medium/long ends: x0.4, x1, x2.5 of their own value
  same          per family, all 3 instances start at the family's middle instance (no diversity at the start)
  random        every period drawn uniformly in [3, 55], seed 7 (one fixed draw)
  short_frozen  short, frozen (indicator LR and gradient x1e-9, ADAPTIVE_INDICATORS false)
  long_frozen   long, frozen
Reading: if learned variants end close to each other while frozen ones differ, learning finds its own periods;
if the starting set matters a lot even when learned, the learning moves the periods little (cf. H16: ~10% of the
indicator gradient comes from direction)."""
import copy, os, random, sys
import yaml
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/ind"); sys.path.insert(0, "runs/tactical/hc4"); sys.path.insert(0, "runs/tactical")
import make_ind as I

ALL = dict(I.FAMS_CLOSE); ALL.update(I.DEFAULT_OHLCV)
clip = lambda v: int(min(60, max(2, round(v))))


def scale_inst(inst, f):
    if isinstance(inst, dict):
        return {k: clip(v * f) for k, v in inst.items()}
    return clip(inst * f)


def mapped(fn):
    return {fam: [fn(fam, i, inst) for i, inst in enumerate(insts)] for fam, insts in ALL.items()}


rng = random.Random(7)
FROZEN = {"INDICATOR_LR_MULT": 1e-9, "INDICATOR_GRAD_MULT": 1e-9, "ADAPTIVE_INDICATORS": False}
sets = {
    "textbook": ({}, None),
    "short": (mapped(lambda f, i, x: scale_inst(x, 0.5)), None),
    "long": (mapped(lambda f, i, x: scale_inst(x, 2.0)), None),
    "wide": (mapped(lambda f, i, x: scale_inst(x, (0.4, 1.0, 2.5)[i])), None),
    "same": ({fam: [copy.deepcopy(insts[1])] * 3 for fam, insts in ALL.items()}, None),
    "random": (mapped(lambda f, i, x: ({k: rng.randint(3, 55) for k in x} if isinstance(x, dict) else rng.randint(3, 55))), None),
    "short_frozen": (mapped(lambda f, i, x: scale_inst(x, 0.5)), FROZEN),
    "long_frozen": (mapped(lambda f, i, x: scale_inst(x, 2.0)), FROZEN),
}
names = []
for name, (fams, extra) in sets.items():
    o = {}
    if fams:
        o["INDICATOR_FAMILIES"] = fams
    if extra:
        o.update(extra)
    spec_name = f"ind2_{name}"
    ov = dict(I.BASE); ov.update(o)
    spec = {"schema_version": 1, "name": spec_name, "base_config": "../default.yaml", "overrides": ov, "slices": I.H.CLIMB,
            "seeds": [0, 1], "run": {"calibrate": False, "epochs": 8, "save_predictions": True}, "rules": I.RULES}
    yaml.safe_dump(spec, open(f"configs/tactical/{spec_name}.yaml", "w"), sort_keys=False)
    names.append(spec_name)
with open("runs/tactical/ind/tasks_ind2.txt", "w") as f:
    for n in names:
        for i in range(2):
            f.write(f"{n} {i}/2\n")
print(len(names), "variants:", " ".join(names))
