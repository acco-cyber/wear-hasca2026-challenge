# waits for the full fit (PID 24828) to finish, stops that process BY ID, then runs folds 0-3 as 4 single-thread
# processes and fold 4 with 4 threads (never kills anything by name)
$log = "E:\Claude code\wear\exp\aka\logs\aka_run.out"
while (-not (Select-String -Path $log -Pattern "fold full\] done" -Quiet)) { Start-Sleep -Seconds 5 }
try { Stop-Process -Id 24828 -Confirm:$false -ErrorAction Stop } catch {}
Set-Location "E:\Claude code\wear\exp\aka"
$env:PYTHONPATH = "E:\Claude code\wear\shim"; $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$ps = foreach ($k in 0..3) { Start-Process -FilePath python -ArgumentList "aka_train.py $k --tag aka --n_est 200 --lr 0.28 --nj 1" -RedirectStandardOutput "logs\f$k.out" -RedirectStandardError "logs\f$k.err" -NoNewWindow -PassThru }
"started " + ($ps | ForEach-Object { $_.Id }) -join " " | Out-File "logs\orch.txt"
$ps | Wait-Process
$env:OMP_NUM_THREADS = "4"; $env:OPENBLAS_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
$p4 = Start-Process -FilePath python -ArgumentList "aka_train.py 4 --tag aka --n_est 200 --lr 0.28 --nj 4" -RedirectStandardOutput "logs\f4.out" -RedirectStandardError "logs\f4.err" -NoNewWindow -PassThru
"fold4 " + $p4.Id | Out-File "logs\orch.txt" -Append
$p4 | Wait-Process
"ALL DONE" | Out-File "logs\orch.txt" -Append
