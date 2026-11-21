# Paper tables, regenerated

Built by `python paper/make_tables.py` from `paper/results/*.json`. Every value is checked against the manuscript by `paper/check_claims.py`.

### Table II -- Principal coverage and reliability results

| Evaluation | BioSR RL-100 | BioSR RL-15 | BioSR TV-100 | BioSR TV-15 | W2S RL-100 | W2S RL-15 |
|---|---|---|---|---|---|---|
| In-domain coverage | 0.899 | 0.895 | 0.895 | 0.887 | 0.891 | 0.867 |
| Zernike-shift coverage | 0.755 | 0.757 | 0.754 | 0.760 | 0.604 | 0.593 |
| Physical-shift coverage | 0.922 | 0.898 | 0.909 | 0.906 | 0.915 | 0.897 |
| One-sided bound validity | 16/16 cells (sd 0.0) on 10 partitions | 16/16 cells (sd 0.0) on 10 partitions | 16/16 cells (sd 0.0) on 10 partitions | 16/16 cells (sd 0.0) on 10 partitions | 6/6 cells (sd 0.0) on 10 partitions | 6/6 cells (sd 0.0) on 10 partitions |
| ρ(coverage, −d⁺) | 0.952 ± 0.025 | 0.952 ± 0.025 | 0.952 ± 0.025 | 0.952 ± 0.025 | 0.638 ± 0.145 | 0.638 ± 0.145 |
| Mean bound slack | 0.046 | 0.046 | 0.046 | 0.046 | 0.052 | 0.052 |
| Real-data coverage (RL) | 0.950 ± 0.017 | 0.950 ± 0.017 | 0.950 ± 0.017 | 0.950 ± 0.017 | 0.891 ± 0.017 | 0.891 ± 0.017 |

### Supp. -- Coverage-level audit, RL-100 (alpha-independent)

| Quantity | 0.1 | 0.05 | 0.02 | 0.01 | 0.005 | 0.001 |
|---|---|---|---|---|---|---|
| Nominal coverage | 0.900 | 0.950 | 0.980 | 0.990 | 0.995 | 0.999 |
| Empirical coverage | 0.901 | 0.948 | 0.983 | 0.993 | 0.997 | 0.998 |
| Mean bound width | 0.4055 | 0.4402 | 0.5145 | 0.5533 | 0.6660 | 0.7309 |
| RL-100: scale floor / violations | 0.00310 | 0 |  |  |  |  |
| RL-15: scale floor / violations | 0.00235 | 0 |  |  |  |  |
| TV-100: scale floor / violations | 0.00223 | 0 |  |  |  |  |
| TV-15: scale floor / violations | 0.00219 | 0 |  |  |  |  |

### Supp. -- Family-resolved Zernike stress test, RL at 100 photons

| Family | Severe err/nom | Severe resid/nom | Severe cov | Extreme err/nom | Extreme resid/nom | Extreme cov |
|---|---|---|---|---|---|---|
| Defocus | 2.77 | 0.93 | 0.829 | 5.00 | 0.86 | 0.486 |
| Astigmatism | 2.49 | 0.96 | 0.686 | 4.48 | 0.90 | 0.429 |
| Coma | 2.84 | 0.92 | 0.771 | 4.89 | 0.84 | 0.314 |
| Spherical | 2.90 | 0.88 | 0.971 | 4.84 | 0.78 | 0.829 |
| Pooled coverage |  |  | 0.814 |  |  | 0.514 |

### Supp. -- Repeated 90% coverage across settings (5 seeds)

| Configuration | In domain | Zernike shift | Physical shift |
|---|---|---|---|
| BioSR RL-100 | 0.899 | 0.755 | 0.922 |
| BioSR RL-15 | 0.895 | 0.757 | 0.898 |
| BioSR TV-100 | 0.895 | 0.754 | 0.909 |
| BioSR TV-15 | 0.887 | 0.760 | 0.906 |
| W2S RL-100 | 0.891 | 0.604 | 0.915 |
| W2S RL-15 | 0.867 | 0.593 | 0.897 |

### Supp. -- Coverage at the largest physical perturbation

| Configuration | NA endpoint | Wavelength endpoint | Focus, 200 nm |
|---|---|---|---|
| BioSR RL-100 | 0.943 | 0.914 | 0.886 |
| BioSR RL-15 | 0.914 | 0.914 | 0.829 |
| BioSR TV-100 | 0.943 | 0.914 | 0.914 |
| BioSR TV-15 | 0.971 | 0.914 | 0.857 |
| W2S RL-100 | 0.896 | 0.896 | 0.771 |
| W2S RL-15 | 0.958 | 0.917 | 0.813 |

### Supp. -- Coverage under 1-SSIM and relative l2

| Configuration | In 1-SSIM | In rel l2 | Zernike 1-SSIM | Zernike rel l2 | Physical 1-SSIM | Physical rel l2 |
|---|---|---|---|---|---|---|
| BioSR RL-100 | 0.899 ± 0.014 | 0.894 ± 0.006 | 0.755 ± 0.024 | 0.614 ± 0.023 | 0.922 ± 0.014 | 0.903 ± 0.009 |
| BioSR RL-15 | 0.895 ± 0.014 | 0.866 ± 0.004 | 0.757 ± 0.022 | 0.581 ± 0.043 | 0.898 ± 0.018 | 0.849 ± 0.019 |
| BioSR TV-100 | 0.895 ± 0.007 | 0.903 ± 0.012 | 0.754 ± 0.034 | 0.704 ± 0.018 | 0.909 ± 0.017 | 0.925 ± 0.012 |
| BioSR TV-15 | 0.887 ± 0.019 | 0.889 ± 0.008 | 0.760 ± 0.050 | 0.667 ± 0.020 | 0.906 ± 0.021 | 0.923 ± 0.014 |
| W2S RL-100 | 0.891 ± 0.011 | 0.927 ± 0.006 | 0.604 ± 0.017 | 0.754 ± 0.024 | 0.915 ± 0.015 | 0.956 ± 0.009 |
| W2S RL-15 | 0.867 ± 0.008 | 0.923 ± 0.003 | 0.593 ± 0.037 | 0.794 ± 0.022 | 0.897 ± 0.009 | 0.966 ± 0.011 |

### Supp. -- Repeated one-sided coverage-bound validation

| Evaluation | Cells | Valid cells (mean over 10 splits) | rho(coverage, -d+) | Mean slack |
|---|---|---|---|---|
| BioSR simulation | 16 | 16.0/16 (sd 0.0) | 0.952 ± 0.025 | 0.046 ± 0.006 |
| W2S simulation | 6 | 6.0/6 (sd 0.0) | 0.638 ± 0.145 | 0.052 ± 0.019 |
| BioSR real grid | 16 | 16.0/16 (sd 0.0) | 0.933 ± 0.053 | 0.088 ± 0.010 |
| W2S genuine widefield | 3 | 3.0/3 (sd 0.0) | -- | 0.050 ± 0.013 |

### Supp. -- Matched-coverage bound-width ablation (5 seeds)

| Setting | All 19 | Residual | Reduction |
|---|---|---|---|
| RL-100 | 0.4015 | 0.5658 | 29.0% |
| RL-15 | 0.4738 | 0.6230 | 24.0% |
| TV-100 | 0.3856 | 0.5354 | 28.0% |
| TV-15 | 0.4381 | 0.5537 | 20.9% |

### Supp. -- Reference-free monitoring on genuine BioSR measurements

| Indicator | Mean rho | SD |
|---|---|---|
| Conditional predictor | 0.222 | 0.153 |
| Physics-consistency profile | 0.197 | 0.055 |
| OTF feature set | 0.182 | 0.041 |
| All observables | 0.166 | 0.044 |
| OTF core | -0.123 | 0.067 |
| Nominal residual | -0.162 | 0.084 |
| Composite blind score | -0.310 | 0.092 |

### Supp. -- In-domain 90% coverage on real measurements (10 seeds)

| Dataset / reconstructor | n_test | Median 1-SSIM | Coverage |
|---|---|---|---|
| BioSR / RL | 137 | 0.368 | 0.950 ± 0.017 |
| BioSR / TV | 137 | 0.374 | 0.966 ± 0.010 |
| BioSR / residual U-Net | 137 | 0.358 | 0.915 ± 0.028 |
| BioSR / unrolled | 137 | 0.354 | 0.958 ± 0.013 |
| W2S / RL | 288 | 0.593 | 0.891 ± 0.017 |
