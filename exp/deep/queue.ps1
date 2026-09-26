param([string]$Name, [string]$JobList)
$Jobs = $JobList -split ";"
# Runs deep.py jobs sequentially; each job is a quoted argument string like "imu f1 --epochs 10 --threads 3".
$env:PYTHONPATH = "E:\Claude code\wear\shim"
$d = "E:\Claude code\wear\exp\deep"
foreach ($j in $Jobs) {
    $tag = ($j -split " ")[0..1] -join "_"
    $p = Start-Process python -ArgumentList "`"$d\deep.py`" $j" -RedirectStandardOutput "$d\logs\q_$tag.log" -RedirectStandardError "$d\logs\q_$tag.err" -WindowStyle Hidden -PassThru
    "$Name $tag PID $($p.Id) $(Get-Date)" | Add-Content "$d\logs\pids.txt"
    $p.WaitForExit()
}
"$Name queue done $(Get-Date)" | Add-Content "$d\logs\pids.txt"
