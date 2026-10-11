# phys7b: primary pair, seeds 1 and 2, 2 shards per cell in parallel, below-normal priority. Resumable.
$env:PYTHONPATH = "D:/nt/nt_phys7b/src"; $env:PYTHONIOENCODING = "utf-8"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"; $env:NT_GPU_MEMORY_LIMIT_MB = "4000"
Set-Location D:/nt/nt_phys7b
$py = "C:/Users/Step/miniforge3/envs/nt/python.exe"
foreach ($cell in "phys7b_none_on", "phys7b_none_off") {
  $ps = @()
  foreach ($sh in "0/2", "1/2") {
    $tag = $sh.Replace("/", "of")
    $p = Start-Process -FilePath $py -ArgumentList "-m neural_trade.cli screen configs/tactical/$cell.yaml --store runs/tactical --shard $sh" `
      -RedirectStandardOutput "runs/tactical/phys7b/${cell}_${tag}.log" -RedirectStandardError "runs/tactical/phys7b/${cell}_${tag}.err" -PassThru -WindowStyle Hidden
    try { $p.PriorityClass = "BelowNormal" } catch {}
    $ps += $p
  }
  $ps | Wait-Process
  "$(Get-Date -Format T) done $cell" | Add-Content runs/tactical/phys7b/progress.txt
}
"all done" | Add-Content runs/tactical/phys7b/progress.txt
