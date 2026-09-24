"""Average available seed outputs into nom_full/test.npy (null-biased, normalised) + nom_full/test_nobias.npy."""
import glob, os, numpy as np
R = r"E:\Claude code\wear\exp\nom\nom_full"
seeds = sorted(d for d in glob.glob(R + r"\s*") if os.path.exists(os.path.join(d, 'test_nobias.npy')))
print('seeds', seeds)
pp = np.mean([np.load(os.path.join(d, 'test_pooled.npy')) for d in seeds], 0)
pt = np.mean([np.load(os.path.join(d, 'test_temporal.npy')) for d in seeds], 0)
c = 0.6 * pt + 0.4 * pp
cb = c.copy(); cb[:, 0] *= np.exp(0.75); cb /= cb.sum(1, keepdims=True)
np.save(os.path.join(R, 'test_pooled.npy'), pp); np.save(os.path.join(R, 'test_temporal.npy'), pt)
np.save(os.path.join(R, 'test_nobias.npy'), c); np.save(os.path.join(R, 'test.npy'), cb.astype(np.float32))
print('saved', cb.shape, np.bincount(cb.argmax(1), minlength=19).tolist())
