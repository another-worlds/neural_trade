"""v3 probes: v2's probe2 (same 14 slices, same readouts) on checkpoints given as NAME=dir. usage: python probe3.py --ckpts c30ref=DIR,c_s0_30k=DIR,... [--results results.jsonl]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.join(HERE, "..", "v2")
sys.path.insert(0, V2); sys.path.insert(0, os.path.join(V2, "..")); sys.path.insert(0, os.path.join(V2, "..", "..", "lab"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import probe2  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", required=True); ap.add_argument("--results", default=HERE + "/results_probe.jsonl")
a = ap.parse_args(); MAP = {k: os.path.abspath(v) for k, v in (x.split("=") for x in a.ckpts.split(","))}


def load_enc(name):
    from channels import N_CH, Standardiser, CTX
    import numpy as np, model2 as M2
    d = MAP[name]; meta = json.load(open(d + "/meta.json"))
    std = Standardiser(np.array(meta["std"]["mean"]), np.array(meta["std"]["std"]))
    enc = M2.Encoder2(N_CH); enc(np.zeros((2, CTX, N_CH), np.float32)); enc.load_weights(d + "/enc.h5")
    return enc, std, meta


probe2.load_enc = load_enc
sys.argv = ["probe2", "--variants", ",".join(MAP), "--results", a.results]
probe2.main()
