import sys
sys.path.insert(0, "D:/nt/nt_wt_124/src"); sys.path.insert(0, "D:/nt/nt_wt_124/tests")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import test_direction_signal_surfaces as T
ex = T._explorer_with_saved({"h0": "ok", "h1": "upper", "h2": "ok"})
ex.refit("none", shrink_delta=False)
both = ex.comparison_table()
print(both.dtypes.to_string())
sty = ex.comparison_table(styled=True)
open("D:/nt/nt124r2_logs/table.html", "w", encoding="utf-8").write(sty.to_html())
cells = [[("n/a" if v != v else (f"{v:.4f}" if isinstance(v, float) else str(v))) for v in row] for row in both.round(4).itertuples(index=False)]
fig, ax = plt.subplots(figsize=(26, 3.2)); ax.axis("off")
tb = ax.table(cellText=cells, colLabels=list(both.columns), rowLabels=[f"{a} {b}" for a, b in both.index], loc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8); ax.set_title(sty.caption or "", fontsize=9)
fig.savefig("D:/nt/nt124r2_logs/table.png", dpi=110, bbox_inches="tight")
print(ex.figures("h1")[0].layout.title.text.replace("<br>", "\n"))
