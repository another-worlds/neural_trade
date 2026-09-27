"""Count test functions whose source touches a window-shaped API (AST, read-only)."""
import ast
import pathlib
import re

ROOT = pathlib.Path("C:/Users/Step/Documents/neural_trade")
PAT = re.compile(r"LOOKBACK|lookback|make_sequences_with_extended_trends|make_inference_windows|windows_test|"
                 r"windows_cal|X_raw|x_window|split_arrays|sequence_anchor_bars|make_purged_splits|WindowNormalizer|"
                 r"applied_periods|_trailing_return_features|interval_scale|lag_features|windows=|predict_last|"
                 r"predict_frame|predict_windows_frame|X_test_seq|test_windows_raw|realised_vol_scale|"
                 r"learnable_indicators|LearnableIndicators|build_gru_attention|Models\.build|build_model|"
                 r"custom_model_factory|EnergyGate|energy_gate|hyper_decoherence")
total_tests = 0
hits = {}
for f in sorted((ROOT / "tests").rglob("test_*.py")):
    src = f.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            total_tests += 1
            seg = ast.get_source_segment(src, node) or ""
            if PAT.search(seg):
                names.append(node.name)
    if names:
        hits[str(f.relative_to(ROOT))] = names
n_hit = sum(len(v) for v in hits.values())
print(f"test functions (def test_*, before parametrisation): {total_tests}; touching window APIs directly: {n_hit}")
for k, v in hits.items():
    print(f"  {k}: {len(v)}")
