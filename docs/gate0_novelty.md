# Gate 0 -- novelty audit

Status: **complete.** Both blocking items resolved; the gap survives. The gap survives, but
it is narrower than the original plan assumed, and there is a specific, named
competitive risk.

Every citation below was retrieved, not recalled. Items marked *unverified* were
not accessible and must not be relied on.

## Verdict

The combination this project targets -- **calibrated, distribution-free,
per-image error bounds for microscopy reconstruction under forward-operator
(PSF) shift** -- remains open. But three separate literatures each own a piece of
the obvious framing, and one research orbit holds two of the three at once.

## Piece 1: calibrated UQ for imaging inverse problems (crowded)

| Paper | What it does | Operator shift? |
|---|---|---|
| Wen, Ahmad, Schniter, *Conformal Bounds on Full-Reference Image Quality*, TMLR 2025 ([2505.09528](https://arxiv.org/abs/2505.09528)) | Conformal bounds on SSIM/PSNR-type metrics; **requires posterior sampling**; denoising + accelerated MRI | No |
| Wen, Ahmad, Schniter, *Task-Driven UQ*, ECCV 2024 ([2405.18527](https://arxiv.org/abs/2405.18527)) | Conformal intervals on a downstream task output; locally adaptive; MRI | No |
| Wen, Ahmad, Schniter, *Minimax Multi-Target CP* ([2511.13533](https://arxiv.org/abs/2511.13533)) | Joint marginal coverage over several targets at once | No |
| Ye, Li, King, Monakhova, *QUTCC* ([2507.14760](https://arxiv.org/abs/2507.14760)) | Quantile-regression U-Net + spatially adaptive conformal calibration; pixel-marginal coverage. **Includes quantitative phase microscopy** | No |
| Ekmekci & Cetin ([2504.07696](https://arxiv.org/abs/2504.07696)) | Conformalised generative Bayesian imaging; aleatoric + epistemic; MRI/CT/inpainting | No |
| Cheung et al., UNSURE @ MICCAI 2025 ([2404.15274](https://arxiv.org/abs/2404.15274)) | Conformal bounds on clinical metrics rather than pixels; sparse-view CT | Partly -- black-box reconstructor failure, not operator shift |
| Silva-Rodriguez & Konukoglu, *Trustworthy Endoscopic SR* ([2604.18001](https://arxiv.org/abs/2604.18001)) | **Conformal Failure Masks** via conformal risk control; lightweight error-prediction net; model-agnostic | No |
| Li, Rosellon-Inclan, Kutyniok, **Starck**, *CHEM* ([2512.09806](https://arxiv.org/abs/2512.09806)) | Wavelet/shearlet hallucination localisation + conformalised quantile regression. Astronomical deconvolution (CANDELS) and natural-image SR (DIV2K); **no microscopy** | **No** -- verified: operator mismatch, PSF error and model misspecification are not addressed anywhere, and it studies no distribution shift |
| Aldridge, **Liaudat**, Pereyra, McEwen ([2605.18655](https://arxiv.org/abs/2605.18655)) | Self-supervised CP + equivariant bootstrapping, **no ground-truth calibration**, motivated by "hidden biases due to distribution shifts between calibration and observed data"; weak-lensing mass mapping | Distribution shift, but not *operator* shift |
| Self-supervised CP for Poisson imaging ([2502.19194](https://arxiv.org/abs/2502.19194), [2502.05127](https://arxiv.org/abs/2502.05127)) | Conformal UQ without clean calibration targets, Poisson noise | No |

A targeted search for PSF misspecification combined with conformal coverage
returned only classical microscopy-PSF literature and no statistical-coverage
work: **the intersection is empty.**

## Piece 2: the null-space theory (already published -- cite, do not claim)

Bhadra, Kelkar, Brooks & Anastasio, *On Hallucinations in Tomographic Image
Reconstruction*, IEEE TMI 40(11) 3249-3260 (2021)
([PMC8673588](https://pmc.ncbi.nlm.nih.gov/articles/PMC8673588/)). Definition 1
decomposes the estimate into measurement and null-space components via explicit
projection operators; Definitions 2-3 give hallucination maps for each; the text
states the null component "cannot be stably estimated from the measurement data
alone".

Their maps require ground truth, so *no-reference estimation* stays open -- but
the theory is theirs.

Related: NanoJ-SQUIRREL (Culley et al., Nature Methods 2018,
[preprint](https://www.biorxiv.org/content/10.1101/158279v1.full)) already
reconvolves a super-resolution image with an **optimised** resolution-scaling
function and compares against a diffraction-limited reference -- structurally the
proposed `min_k r_k` over an operator family. It gives no statistical guarantee,
needs a separately acquired reference image, and states it cannot detect errors
smaller than the diffraction limit, which is precisely the blindness this project
measures.

## Piece 3: PSF error propagation (the blind spot -- astronomy owns it)

The literature the original plan missed entirely, and it is mature.

* **Paulin-Henriksson et al. (2008)** pioneered PSF error propagation for weak
  lensing; **Massey et al. (2012)** extended it. The framework propagates PSF
  model error into second-order image moments (ellipticity and size).
* **Liaudat, Starck & Kilbinger**, *Point spread function modelling for
  astronomical telescopes: a review focused on weak gravitational lensing
  studies*, Frontiers in Astronomy and Space Sciences (2023)
  ([2306.07996](https://arxiv.org/abs/2306.07996)) -- retrieved. A full review of
  PSF modelling and validation metrics. Does **not** mention conformal
  prediction or distribution-free coverage.
* *Point spread function errors for weak lensing-density cross-correlations*,
  A&A (2024) -- PSF error propagation on a real survey.
* Radio interferometry has the same problem in worse form: the instrument
  response is a major uncertainty source because an interferometer PSF is
  "typically poorly behaved", making reconstruction more sensitive to PSF model
  error even when the response can be forward modelled.
* **Calibration marginalisation** -- carrying calibration-parameter uncertainty
  into the inference instead of fixing the parameters -- is standard practice in
  cosmology. That is the astronomy analogue of the proposed PSF uncertainty set,
  so a candidate-operator family is not a novel idea either.

### The competitive risk, stated plainly

**Starck** co-authors CHEM (conformal + hallucination quantification).
**Liaudat** co-authors both the astronomy PSF-modelling review and the
self-supervised conformal imaging paper. One research orbit therefore holds PSF
modelling expertise, conformal imaging UQ, and hallucination quantification at
the same time. They are one step from this project, in the astronomy domain.

This does not kill the project, but it fixes what the wedge has to be.

## Consequences for the paper

1. **Cite the astronomy lineage** -- Paulin-Henriksson (2008), Massey (2012),
   Liaudat et al. (2023). Otherwise a reviewer from that community reads the
   paper as naive about a 15-year-old problem.
2. **The defensible differentiators, all three needed together:**
   * *Per-image, finite-sample, distribution-free* bounds. Astronomy propagates
     PSF error into population-level summary statistics with analytic moment
     error budgets, not per-image coverage.
   * *Operator shift as the calibration covariate.* Every conformal imaging paper
     above assumes a known, fixed operator.
   * *Microscopy*, where the PSF is estimated, drifts, and varies with wavelength
     and immersion, and where the null-space band is large -- measured here at
     11.6% of object energy for a faithful operator choice.
3. **Do not claim** the null-space decomposition, the reconvolution-residual
   idea, or kernel-weighted conformal calibration. All three are published.
4. **Move fast on the phenomenon result.** Gates 1 and 2 already yield a clean,
   quantitative finding -- the residual sits at the photon noise floor and the
   whole mismatch signal is a few percent of it. That stands on its own even if
   the calibration contribution is scooped.

## Resolved since the first pass

* **CHEM** ([2512.09806](https://arxiv.org/abs/2512.09806)) full text checked. It is
  the nearest competitor *in the deconvolution setting* -- conformalised quantile
  regression over wavelet/shearlet hallucination maps, applied to astronomical
  deconvolution -- but the PSF is known and fixed. Forward-operator mismatch, PSF
  error and model misspecification are not addressed, no distribution shift is
  studied, hallucination is not defined via the operator null space, and there is
  no microscopy. **The gap survives it.** Cite as closest prior art for
  "conformal + hallucination + deconvolution".
* **BayesDL-SIM** -- Liu T, Liu J, Li D, Tan S, *Bayesian deep-learning structured
  illumination microscopy enables reliable super-resolution imaging with
  uncertainty quantification*, Nature Communications (2025). Retrieved via Europe
  PMC after two reCAPTCHA blocks on PMC. It provides Bayesian uncertainty for SIM
  and "identif[ies] erroneous generalizations in various model misuse scenarios",
  but the abstract shows **no calibration or coverage validation** of that
  uncertainty and **no PSF or optical-model mismatch**. So it is the closest
  microscopy-UQ work and the natural baseline to compare against, and it is not
  a calibrated-coverage method. **The gap survives it.**

## Still outstanding (non-blocking)

* Read BayesDL-SIM in full to confirm what "model misuse scenarios" covers -- if
  it includes optical mismatch, it becomes a direct baseline rather than an
  adjacent one.
* Motivation citation found but not yet read: *Point spread function decoupling in
  computational fluorescence microscopy*, Light: Science & Applications (2025),
  `s41377-025-02112-5`.
