param([string]$name, [string]$script, [string]$argstr)
$env:OMP_NUM_THREADS = "4"; $env:OPENBLAS_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
$log = "E:\Claude code\wear\exp\base\logs\$name.log"
New-Item -ItemType Directory -Force "E:\Claude code\wear\exp\base\logs" | Out-Null
$p = Start-Process -FilePath "C:\Users\Koushik\AppData\Local\Programs\Python\Python311\python.exe" -ArgumentList "-u `"E:\Claude code\wear\exp\base\$script`" $argstr" -WorkingDirectory "E:\Claude code\wear\exp\base" -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
"PID $($p.Id) -> $log"
