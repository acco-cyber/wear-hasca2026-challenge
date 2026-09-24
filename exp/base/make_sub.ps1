# Writes a test submission from exp\base\<model>\test.npy with the lead's current decoder
# (null_scale 0.5 before graph smoothing, graph k=10 alpha 0.5 iters 5, chains thr -6, count band 80-250, p_stay 0.8).
# usage: powershell -File make_sub.ps1 best            -> E:\Claude code\wear\subs\sub_base_best.csv
param([string]$model = "best")
$env:OMP_NUM_THREADS = "4"
Set-Location "E:\Claude code\wear\src"
python predict.py --probs "E:\Claude code\wear\exp\base\$model\test.npy" --mode graph_viterbi --k 10 --alpha 0.5 --iters 5 --thr=-6 --p_stay 0.8 --calib --lo 80 --hi 250 --null_scale 0.5 --out "sub_base_$model.csv"
