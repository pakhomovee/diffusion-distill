"""EDM sigma schedule, DSM training loop, and the deterministic sampler."""
import torch


def sample_sigma_train(n, P_mean=-1.2, P_std=1.2, device="cpu"):
    return (torch.randn(n, device=device) * P_std + P_mean).exp()


def edm_sigmas(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    i = torch.arange(n_steps, device=device, dtype=torch.float64)
    a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    s = (a + i / max(n_steps - 1, 1) * (b - a)) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64, device=device)]).float()


def dsm_loss(model, x0, sigma_data=1.0):
    sig = sample_sigma_train(x0.shape[0], device=x0.device)
    n = torch.randn_like(x0) * sig[:, None]
    D = model(x0 + n, sig)
    w = (sig ** 2 + sigma_data ** 2) / (sig * sigma_data) ** 2
    return (w[:, None] * (D - x0) ** 2).mean()


def train_teacher(model, sampler, steps=4000, bs=512, lr=2e-3, sigma_data=1.0,
                  log_every=500, ema_decay=0.999, verbose=True):
    import copy
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    hist = []
    for it in range(steps):
        x0 = sampler(bs)
        loss = dsm_loss(model, x0, sigma_data)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - ema_decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        hist.append(loss.item())
        if verbose and (it + 1) % log_every == 0:
            print(f"    it {it+1:5d}  dsm {sum(hist[-log_every:])/log_every:.4f}", flush=True)
    return ema, hist


@torch.no_grad()
def heun_sample(model, n, d, n_steps=32, device="cpu", z=None, sigmas=None):
    sig = edm_sigmas(n_steps, device=device) if sigmas is None else sigmas
    x = (torch.randn(n, d, device=device) if z is None else z) * sig[0]
    for i in range(len(sig) - 1):
        s, s1 = sig[i], sig[i + 1]
        D = model(x, s.expand(x.shape[0]))
        dxt = (x - D) / s
        xn = x + (s1 - s) * dxt
        if s1 > 0:
            D2 = model(xn, s1.expand(x.shape[0]))
            dxt2 = (xn - D2) / s1
            xn = x + (s1 - s) * 0.5 * (dxt + dxt2)
        x = xn
    return x
