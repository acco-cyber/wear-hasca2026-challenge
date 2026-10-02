param([string]$links = "L2", [string]$set = "main")
$env:PYTHONPATH = "E:\Claude code\wear\shim"
Set-Location "E:\Claude code\wear\exp\hyb\agents\l2oof"
$lp = "E:\Claude code\wear\work\hanbat\cv_base_oof_logp_b.npy"
$lab = "E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl"
if ($set -eq "main") {
  $cfgs = @(
    @("a", ""),
    @("b", "--prior 0.3 --counts 0.3 --gate 0.55:top2"),
    @("c", "--prior 0.3 --counts 0.3 --gate 0.55:top2 --extra_links"),
    @("d", "--prior 0.3 --counts 0.3 --gate 0.5:top2 --extra_links --xl_cand 2")
  )
} else {
  $cfgs = @(
    @("c_p0.2", "--prior 0.2 --counts 0.3 --gate 0.55:top2 --extra_links"),
    @("c_p0.4", "--prior 0.4 --counts 0.3 --gate 0.55:top2 --extra_links"),
    @("c_c0.2", "--prior 0.3 --counts 0.2 --gate 0.55:top2 --extra_links"),
    @("c_c0.5", "--prior 0.3 --counts 0.5 --gate 0.55:top2 --extra_links"),
    @("c_g0.5", "--prior 0.3 --counts 0.3 --gate 0.5:top2 --extra_links"),
    @("c_g0.6", "--prior 0.3 --counts 0.3 --gate 0.6:top2 --extra_links"),
    @("c_xw0.5", "--prior 0.3 --counts 0.3 --gate 0.55:top2 --extra_links --xl_w 0.5"),
    @("c_xw1.5", "--prior 0.3 --counts 0.3 --gate 0.55:top2 --extra_links --xl_w 1.5")
  )
}
foreach ($c in $cfgs) {
  $args_ = @("graph_lab_l2.py", "cv", $lp, $lab, "--links", $links, "--tag", "$($c[0])_$links") + ($c[1] -split " " | Where-Object { $_ -ne "" })
  & python @args_
}
