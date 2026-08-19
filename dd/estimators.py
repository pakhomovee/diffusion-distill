"""The two score estimators and the doubly-robust combination weight.

A(x) = teacher score            : deterministic given x  -> zero variance, bias b_A
B(x) = minibatch empirical score: the exact score of the noised empirical
       measure of a real minibatch. This is the nonparametric estimator that is
       actually available at an off-distribution (student) x, and it is what a
       DMD2/ADD/LADD discriminator learns to approximate.

MSE of s_hat = lam*A + (1-lam)*B, with u = b_A - b_B (A has no variance):
    MSE(lam) = ||lam b_A + (1-lam) b_B||^2 + (1-lam)^2 V_B
    lam* = (V_B - <b_B,u>) / (||u||^2 + V_B)            EXACT, needs true score
    lam_hat = V_B / E||A-B||^2                          ONLINE, assumes b_B = 0
(using E||A-B||^2 = ||u||^2 + V_B).
"""
import torch


def empirical_score(x, x0, sigma, chunk=4096):
    """Score of the empirical measure of x0 convolved with N(0, sigma^2 I).

    x:  (B,d) query points.  x0: (N,d) real samples.  sigma: scalar or (B,).
    Returns (B,d).  No (B,N,d) tensor is materialised.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    sig = sigma.reshape(-1)
    if sig.numel() == 1:
        sig = sig.expand(x.shape[0])
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], chunk):
        xb, sb = x[i:i + chunk], sig[i:i + chunk, None]
        d2 = torch.cdist(xb, x0) ** 2                      # (b,N)
        w = torch.softmax(-d2 / (2 * sb ** 2), dim=1)      # (b,N) posterior over data
        out[i:i + chunk] = (w @ x0 - xb) / sb ** 2
    return out


@torch.no_grad()
def lambda_diagnostics(x, sigma, teacher_score, data_sampler, true_score,
                       n_batches=32, batch_n=512):
    """Measure b_A, b_B, V_B and both lambdas at one noise level.

    x: (B,d) evaluation points.  Returns a dict of scalars (aggregated over x).
    """
    A = teacher_score(x, sigma)
    s = true_score(x, sigma)

    Bs = []
    lam_hat_num, lam_hat_den = 0.0, 0.0
    for _ in range(n_batches):
        x0 = data_sampler(batch_n)
        Bfull = empirical_score(x, x0, sigma)
        Bs.append(Bfull)
        # half-batch split -> variance estimate available online
        h = batch_n // 2
        B1 = empirical_score(x, x0[:h], sigma)
        B2 = empirical_score(x, x0[h:], sigma)
        lam_hat_num += ((B1 - B2) ** 2).sum(-1).mean().item() / 4.0
        lam_hat_den += ((A - Bfull) ** 2).sum(-1).mean().item()
    lam_hat_num /= n_batches
    lam_hat_den /= n_batches

    Bst = torch.stack(Bs)                                   # (M,B,d)
    EB = Bst.mean(0)
    V_B = ((Bst - EB) ** 2).sum(-1).mean().item()           # E_x Var
    b_A, b_B = A - s, EB - s
    u = b_A - b_B
    num = V_B - (b_B * u).sum(-1).mean().item()
    den = (u ** 2).sum(-1).mean().item() + V_B
    lam_star = num / max(den, 1e-30)
    lam_hat = lam_hat_num / max(lam_hat_den, 1e-30)

    def mse(l):
        e = (l * b_A + (1 - l) * b_B)
        return (e ** 2).sum(-1).mean().item() + (1 - l) ** 2 * V_B

    sn = (s ** 2).sum(-1).mean().item()                     # for normalisation
    return dict(
        lam_star=lam_star, lam_hat=lam_hat,
        bias_A=(b_A ** 2).sum(-1).mean().item(), bias_B=(b_B ** 2).sum(-1).mean().item(),
        var_B=V_B, score_norm2=sn,
        mse_teacher=mse(1.0), mse_data=mse(0.0),
        mse_star=mse(lam_star), mse_hat=mse(lam_hat),
    )
