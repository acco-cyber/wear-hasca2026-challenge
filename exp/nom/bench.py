import sys, time, torch, torch.nn as nn
sys.argv = [sys.argv[0], '--out', 'smoke', '--wps', '1']
nt = int(__import__('os').environ.get('NT', '4'))
exec(open(r"E:\Claude code\wear\exp\nom\nom_train.py").read().split("pooled = PooledFusionModel()")[0])
torch.set_num_threads(nt)
m = TemporalFusionModel(); opt = torch.optim.AdamW(m.parameters(), 1e-3)
x = torch.randn(256, 4, 3, 50); v = torch.randn(256, 15, 768); yb = torch.randint(0, 19, (1024,))
def step(fn):
    t = time.time()
    for _ in range(5):
        opt.zero_grad(); l = fn(); l.backward(); opt.step()
    return (time.time() - t) / 5
m.train()
print('threads', nt)
print('full step', step(lambda: F.cross_entropy(m(x, v), yb)))
print('imu enc step', step(lambda: m.inertial_encoder(x.reshape(-1, 3, 50)).sum()))
print('vid enc step', step(lambda: m.video_encoder(v).sum()))
t = time.time()
for _ in range(5): recon(torch.randn(256, 15, 160))
print('recon', (time.time() - t) / 5)
