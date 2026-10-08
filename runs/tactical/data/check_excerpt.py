"""The excerpt must give a long-block run exactly the same windows as the full file: compares, per slice, the windowed
arrays the screen trial uses (X_seq, y_seq, last_close_seq, extended_trends, X_model; the 126,000 most recent windows)
built from Bitcoin_BTCUSDT.csv and from Bitcoin_BTCUSDT_tactical.csv, and times both. CPU only."""
import json, sys, time
import numpy as np
import neural_trade  # noqa: F401
from neural_trade.core.config import Config
from neural_trade.experiments import screen as S

slices = sys.argv[1:] or ["2017-05-02T12:00:00", "2020-03-30T04:00:00"]
over = dict(N_FOLDS=2, VAL_FRACTION=0.1, CAL_FRACTION=0.1, MAX_SEQUENCE_COUNT=126000, FOLD_INDEX=-2, BATCH_SIZE=256)
res = {}
for de in slices:
    arrs, secs = {}, {}
    for name, path in (("full", "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"), ("excerpt", "D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv")):
        cfg = Config.from_yaml("configs/default.yaml").override(CSV_PATH=path, DATA_END=de, **over)
        t = time.time(); cache = {}
        S._load_cached(cfg, cache)
        arrs[name] = S._windowed_cached(cfg, cache)
        secs[name] = round(time.time() - t, 1)
        del cache
    eq = [bool(np.array_equal(a, b)) for a, b in zip(arrs["full"], arrs["excerpt"])]
    res[de] = {"identical_arrays": eq, "shapes": [list(np.shape(a)) for a in arrs["excerpt"]], "seconds": secs}
    print(de, res[de], flush=True)
    del arrs
json.dump(res, open("runs/tactical/data/excerpt_check.json", "w"), indent=1)
