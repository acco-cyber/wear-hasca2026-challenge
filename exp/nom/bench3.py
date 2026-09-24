import sys, time, os, torch, torch.nn as nn
sys.argv = [sys.argv[0], '--out', 'smoke', '--wps', '1']
nt = int(os.environ.get('NT', '4'))
exec(open(r"E:\Claude code\wear\exp\nom\nom_train.py").read().split("def combine(pp, pt):")[0])
torch.set_num_threads(nt)
x = torch.randn(256, 4, 3, 50); v = torch.randn(256, 15, 160); yb = torch.randint(0, 19, (1024,))
def step(m, n=3):
    opt = torch.optim.AdamW(m.parameters(), 1e-3); m.train()
    t = time.time()
    for _ in range(n):
        opt.zero_grad(); l = F.cross_entropy(m(x, recon(v)), yb); l.backward(); opt.step()
    return (time.time() - t) / n
print('threads', nt, 'pooled', round(step(PooledFusionModel(), 10), 3), 'temporal', round(step(TemporalFusionModel()), 3), flush=True)
