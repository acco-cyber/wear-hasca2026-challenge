import sys, time, torch, torch.nn as nn
sys.argv = [sys.argv[0], '--out', 'smoke', '--wps', '1']
exec(open(r"E:\Claude code\wear\exp\nom\nom_train.py").read().split("pooled = PooledFusionModel()")[0])
torch.set_num_threads(4)
x = torch.randn(1024, 3, 50)
def step(m, fn, n=4):
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    fn(); t = time.time()
    for _ in range(n):
        opt.zero_grad(); l = fn(); l.backward(); opt.step()
    return (time.time() - t) / n
m = InertialEncoder(); m.train()
print('default', step(m, lambda: m(x).sum()))
torch.backends.mkldnn.enabled = False
print('mkldnn off', step(m, lambda: m(x).sum()))
torch.backends.mkldnn.enabled = True
with torch.autograd.profiler.profile() as prof:
    for _ in range(2):
        l = m(x).sum(); l.backward()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
