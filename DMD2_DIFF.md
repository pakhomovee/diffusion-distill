# Our DMD2 against the reference implementation

Read from `tianweiy/DMD2` at `main/edm/` — the EDM/ImageNet-64 path, which is the
closest analogue to our cheap tier (pixel space, EDM-family teacher). The
published GAN run is
`experiments/imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch.sh`,
and every "DMD2" number below is from that script rather than from an argparse
default — several defaults are *not* what the paper ran.

## What matches, verified line by line

**The distribution-matching gradient is identical.**

```python
# DMD2 main/edm/edm_guidance.py
p_real = latents - pred_real_image
p_fake = latents - pred_fake_image
weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
grad = (p_real - p_fake) / weight_factor
loss = 0.5 * F.mse_loss(original_latents, (original_latents - grad).detach())

# ours ddgpu/dmd2.py:dm_loss
grad = (D_fake - D_real)
norm = (x0.detach() - D_real).abs().mean(dim=(1, 2, 3), keepdim=True)
grad = grad / norm.clamp_min(1e-4)
loss = 0.5 * F.mse_loss(x0, (x0 - grad).detach())
```

`p_real - p_fake = (x0 - D_real) - (x0 - D_fake) = D_fake - D_real`, and the
normaliser is `|x0 - D_real|` in both. Same numerator, same denominator, same
surrogate. We additionally clamp the denominator at 1e-4.

**The critic's DSM weighting is identical.** Theirs `weights = snrs + 1.0 /
sigma_data**2`; ours `dsm_weight = (sigma^2 + sigma_data^2) / (sigma *
sigma_data)^2`, which expands to `1/sigma^2 + 1/sigma_data^2`. Both reduce with
`.mean()` over all elements.

**The discriminator attaches to the same place.** Theirs calls
`self.fake_unet(image, timestep_sigma, label, return_bottleneck=True)` — the
CRITIC's bottleneck. Ours is `DiffusersUNetAdapter.trunk()`, the critic's
mid-block. Same tensor, same network, same reason.

Also matching: generator initialised from the teacher (`--initialie_generator`
/ `init_from_teacher`); separate optimisers for generator and critic; the DM
sigma window is the middle 2%–98% of the schedule on both sides (their
`min_step_percent 0.02 / max_step_percent 0.98`, our `t_min=20 / t_max=979` of
1000); and the discriminator sees noised rather than clean images.

**The two-weight split is real and we now have it.** DMD2 carries
`cls_loss_weight` for the discriminator's OWN objective and
`gen_cls_loss_weight` for its influence on the generator, applied in different
places (`train_edm.py:254` and `:282`). Our `gan_d_weight` / `gan_weight`
mirror that exactly.

## What differs

Both codebases average the diffusion losses over `B*C*H*W` and the adversarial
losses over `B`, so a weight is only meaningful against `N = C*H*W`. Porting a
weight between resolutions means holding `w*N` fixed: ImageNet-64 is
`3*64*64 = 12288`, CIFAR-10 is `3*32*32 = 3072`, a factor of 4.

| | DMD2 (published) | ours (before) | verdict |
|---|---|---|---|
| critic updates per generator update | `dfake_gen_update_ratio 5` | `d_steps 1` | **missing TTUR** |
| generator adversarial loss | `softplus(-logit)` | `-logit.mean()` | **unbounded** |
| discriminator loss | `softplus(fake) + softplus(-real)` | hinge | differs |
| head input | bottleneck, spatial, via convs | bottleneck, **mean-pooled** | **weaker** |
| `gen_cls_loss_weight` | 3e-3, `w*N = 36.9` | 1e-3, `w*N = 3.07` | **12x too weak** |
| `cls_loss_weight` | 1e-2, `w*N = 122.9` | 1.0, `w*N = 3072` | **25x too strong** |
| learning rate | 2e-6 (both) | 1e-5 (both) | 5x higher |
| separate lr for the head | no | added `lr_gan` | diverges |
| batch | 40 x 7 GPUs = 280 | 480-512 x 1 | fine |

### The two that most plausibly explain the degeneration

**`-logit.mean()` is unbounded.** DMD2's `softplus(-logit)` saturates: once the
discriminator is fooled the gradient vanishes. Ours rewards the generator
without limit for driving the logit up, so there is nothing to stop it walking
off into whatever direction the discriminator happens to score highly.

**The head mean-pools.** `GANHead` does `self.out(h.mean(1))`, collapsing the
spatial dimension before the classifier, where DMD2 keeps it through two
strided convs. A mean-pooled discriminator can be satisfied by a global shift
in the feature average — and the failed run produced exactly that: near-black
images with red crushed to 23% of correct and the three channels' diversity
ratios flying apart (0.53 / 0.86 / 1.46).

Together they describe a generator that found a cheap unbounded direction in a
discriminator that only looks at averages.

## Ported settings for CIFAR-10

Holding `w*N` fixed against the published ImageNet-64 run:

    gan_weight    = 3e-3 * 12288/3072 = 1.2e-2
    gan_d_weight  = 1e-2 * 12288/3072 = 4.0e-2
    d_steps       = 5
    lr_g = lr_d   = 2e-6      (theirs; ours has been 1e-5)

`d_steps=5` also raises the step cost: the critic path runs five times per
generator update.

## Still open

* The head architecture. Matching DMD2 means replacing the mean-pool with
  strided convs over the bottleneck's spatial extent, which needs `trunk()` to
  report the spatial shape rather than only tokens. Not done.
* The learning rate. Ours is 5x theirs, and they train for far longer. Left
  alone for now, since changing it at the same time as everything else would
  make the next run uninterpretable.
* `lr_gan`. DMD2 puts the head in the critic's optimiser at the critic's rate
  and relies on `dfake_gen_update_ratio 5` to let the discriminator keep up. Our
  separate rate is a reasonable alternative but is NOT what the reference does,
  so it now defaults to `lr_d` and is opt-in.
