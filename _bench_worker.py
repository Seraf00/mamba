
import json, sys, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
WORKERS, BATCHES, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

class Synth(Dataset):
    def __len__(self): return 1600
    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        x = rng.random((256, 256), dtype=np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = np.rot90(x, i % 4).copy()
        y = (rng.random((256, 256)) * 4).astype(np.int64)
        return torch.from_numpy(x)[None], torch.from_numpy(y)

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, bf=64, n=4):
        super().__init__()
        self.e = nn.ModuleList(); self.d = nn.ModuleList()
        self.u = nn.ModuleList(); ch, f = 1, bf
        for _ in range(n):
            self.e.append(block(ch, f)); ch = f; f *= 2
        self.b = block(ch, f)
        for _ in range(n):
            self.u.append(nn.ConvTranspose2d(f, f // 2, 2, 2))
            self.d.append(block(f, f // 2)); f //= 2
        self.o = nn.Conv2d(f, 4, 1); self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s = []
        for e in self.e:
            x = e(x); s.append(x); x = self.pool(x)
        x = self.b(x)
        for u, d, sk in zip(self.u, self.d, reversed(s)):
            x = d(torch.cat([u(x), sk], 1))
        return self.o(x)

def main():
    dev = torch.device('cuda')
    m = UNet().to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(); scaler = torch.amp.GradScaler('cuda')
    dl = DataLoader(Synth(), batch_size=8, shuffle=True,
                    num_workers=WORKERS, pin_memory=True,
                    **({'prefetch_factor': 2} if WORKERS else {}))
    def step(b):
        x = b[0].to(dev, non_blocking=True); y = b[1].to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = crit(m(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    it = iter(dl)
    for _ in range(3):
        try: step(next(it))
        except StopIteration: it = iter(dl); step(next(it))
    torch.cuda.synchronize()
    t0 = time.perf_counter(); done = 0
    while done < BATCHES:
        try: b = next(it)
        except StopIteration: it = iter(dl); continue
        step(b); done += 1
    torch.cuda.synchronize()
    json.dump({'it_per_s': done / (time.perf_counter() - t0)}, open(OUT, 'w'))

# Required under the spawn start method: workers re-import this module.
if __name__ == '__main__':
    main()
