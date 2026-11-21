# Documentation

## Reproducing the paper

| Document | Use it for |
|---|---|
| [../README.md](../README.md) | One-command reproduction from the shipped caches (level A) |
| [reproduction.md](reproduction.md) | Rebuilding the caches from raw BioSR and W2S data (level B) and training the networks (level C) |
| [../experiments/README.md](../experiments/README.md) | What each shipped cache contains and the exact command that built it |
| [../data/README.md](../data/README.md) | Obtaining the raw data, licences, verified acquisition parameters |
| [colab.md](colab.md) | Running the GPU step on Colab |
| [../paper/results/claims_report.md](../paper/results/claims_report.md) | Every number in the paper next to its regenerated value |

## Research notes

Written while the work was done and kept as a record. They explain why the
method looks the way it does, including results that were later withdrawn.
Numbers in them can predate the final protocol; the paper's numbers are the
ones in `paper/results/`.

| Document | Topic |
|---|---|
| [PROJECT.md](PROJECT.md) | The question, physics, prior art and method, gate by gate |
| [gate0_novelty.md](gate0_novelty.md) | Literature and novelty audit |
| [gate4_results.md](gate4_results.md) | Calibration under shift, first pass |
| [otf_evaluation.md](otf_evaluation.md) | Whether the OTF feature block predicts coverage loss |
| [realism_check.md](realism_check.md) | Simulated versus real widefield noise and spectra |
| [sim2real.md](sim2real.md) | Simulation-to-real transfer and the out-of-focus fraction |
| [gpu_reconstructors.md](gpu_reconstructors.md) | The two learned reconstructors |
| [history/](history/) | Status log, manuscript drafting notes, and the v1.0.0 README narrative |
