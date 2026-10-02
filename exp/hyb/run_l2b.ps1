$env:PYTHONPATH = "E:\Claude code\wear\shim"; $env:OMP_NUM_THREADS = "4"
Set-Location "E:\Claude code\wear"
$L = "work\hanbat\cv_base_oof_logp_b.npy"; $O = "exp\pl\labels_v3bv1f.pkl"
$cfgs = @(
  @("prior0.3", "--prior 0.3"),
  @("counts0.3", "--counts 0.3"),
  @("gate0.55top2", "--gate 0.55:top2"),
  @("gate0.6plain", "--gate 0.6"),
  @("prior0.3+gate0.5top2", "--prior 0.3 --gate 0.5:top2"),
  @("prior0.5+counts0.3+gate0.55", "--prior 0.5 --counts 0.3 --gate 0.55:top2"),
  @("icm", "--prior 0.3 --counts 0.3 --gate 0.55:top2 --icm 5,4.0,0.1")
)
foreach ($c in $cfgs) {
  $args_ = @("exp\hyb\graph_lab.py", "cv", $L, $O, "--links", "L2", "--tag", $c[0]) + ($c[1] -split " " | Where-Object { $_ -ne "" })
  & python @args_ 2>&1 | Select-String "F1"
}
