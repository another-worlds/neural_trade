"""Writes configs/tactical/o2_none_{on,off}.yaml = phys7b configs with seeds [0, 1], run.epochs 8 (SPEC.md).
Run from the repo root: python runs/tactical/outlook/O2_phys_var8/make_o2.py"""
import yaml

for phys in ("on", "off"):
    spec = yaml.safe_load(open(f"configs/tactical/phys7b_none_{phys}.yaml"))
    spec["name"] = f"o2_none_{phys}"
    spec["seeds"] = [0, 1]
    spec["run"]["epochs"] = 8
    yaml.safe_dump(spec, open(f"configs/tactical/o2_none_{phys}.yaml", "w"), sort_keys=False)
    print(spec["name"])
