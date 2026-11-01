# Money blockers: supporting files

Write-up: [`plans/money_blockers.md`](../../plans/money_blockers.md).

| File | Content |
|---|---|
| `critique.json` | 48 blockers from five critics (information, execution, objective, model, strategy), each with evidence, mechanism, fix or test, and the skeptic's verdict (`confirmed` / `partly` / `refuted`), corrected evidence and judgement on whether removing it would create profit. Script paths inside refer to the session scratch directory (`<scratch>/`) and are provenance only. |
| `probes.py` | The lead's probes: volatility regimes vs break-even accuracy, the volatility-gated reversal with time and bracket exits, and Deribit implied vs forward realised volatility. Needs the Timestral-LIG cache and network access. |
| `probes_output.txt` | Output of `probes.py`. |
