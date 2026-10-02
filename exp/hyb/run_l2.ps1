param([string]$set = "replay")
$env:PYTHONPATH = "E:\Claude code\wear\shim"; $env:OMP_NUM_THREADS = "4"
Set-Location "E:\Claude code\wear"
$L = "work\hanbat\cv_base_oof_logp_b.npy"; $O = "exp\pl\labels_v3bv1f.pkl"
$AD = "work\hanbat\ad_gl7_oof.npy"; $ST = "work\hanbat\st_gl7_oof_A.npy"
$R = "--prior 0.3 --counts 0.3 --gate 0.55:top2"
if ($set -eq "replay") {
  $cfgs = @(
    @("hb0  (LB .90581)", ""),
    @("gl6  (LB .91044)", "$R"),
    @("gl7  (LB .91289)", "$R --extra_links"),
    @("gl11 (LB .91367)", "$R --extra_links --st ${ST}:0.08"),
    @("gl12 (LB .91197)", "$R --extra_links --st ${ST}:0.15"),
    @("gl13 (LB .90852)", "$R --extra_links --xlogp ${AD}:0.5 --st ${AD}:0.15"),
    @("gl8b (LB .90957)", "--prior 0.3 --counts 0.3 --gate 0.5:top2 --extra_links --xl_cand 2")
  )
}
foreach ($c in $cfgs) {
  $args_ = @("exp\hyb\graph_lab.py", "cv", $L, $O, "--links", "L2", "--tag", $c[0]) + ($c[1] -split " " | Where-Object { $_ -ne "" })
  & python @args_ 2>&1 | Select-String "F1"
}
