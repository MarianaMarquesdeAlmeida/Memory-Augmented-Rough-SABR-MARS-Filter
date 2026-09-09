# Memory-Augmented Rough-SABR (MARS) Filter

This repository contains the code and analysis notebooks for an MSc thesis on
sequential filtering of Bitcoin option surfaces. The empirical study compares a
Markovian SABR particle filter benchmark with a Memory-Augmented Rough-SABR
particle filter.

The empirical focus is sequential prediction and filtering of Deribit BTC
option panels. The main comparison is based on paired Markovian and Rough-SABR
runs using the same windows, quote sets, evaluation settings, and random seeds.

## Project Layout

```text
.
|-- notebooks/          Curated analysis notebooks
|-- src/                Core particle-filter implementations
|-- data/               Local raw and processed data, not committed
|-- outputs/            Local model outputs and notebook exports, not committed
|-- requirements.txt    Minimal Python dependencies
|-- README.md
```

The `data/` and `outputs/` directories are intentionally ignored because the raw
market data and generated experiment outputs are large.

## Core Models

The repository compares two nonlinear state-space models.

| Model | Description |
|---|---|
| Markovian SABR PF | SABR particle filter benchmark with latent forward, volatility level, correlation, and volatility-of-volatility. |
| MARS / Rough-SABR PF | Memory-augmented Rough-SABR particle filter with finite-dimensional memory factors approximating the rough-volatility kernel. |

The main model files are:

| File | Purpose |
|---|---|
| `src/normal_sabr_pf.py` | Markovian SABR benchmark particle filter. |
| `src/rough_sabr_pf.py` | Memory-Augmented Rough-SABR particle filter. |

## Analysis Notebooks

The notebooks in `notebooks/` are organized as a compact, thesis-facing analysis
suite.

| Notebook | Purpose |
|---|---|
| `chapter2_btc_data_descriptive_plots.ipynb` | Descriptive BTC option-data diagnostics used before the filtering analysis. |
| `predictive_likelihood.ipynb` | Main headline predictive log-likelihood comparison for paired Markovian and Rough-SABR runs. |
| `nonoverlapping_window_analysis.ipynb` | Four non-overlapping window comparison of predictive scores and pricing errors. |
| `predictive_uncertainty_calibration.ipynb` | Predictive interval coverage, interval score, width, and calibration by moneyness. |
| `filtered_state_summaries.ipynb` | Filtered latent-state summaries and state-path diagnostics. |
| `particle_stability_numerical_behaviour.ipynb` | ESS, resampling, tempering, and runtime behaviour across windows and seeds. |
| `rough_kernel_lift_diagnostics.ipynb` | Finite rough-kernel lift nodes, weights, approximation error, and ratio plots. |
| `robustness_q160_eval300_analysis.ipynb` | Robustness checks for the q160/eval300 experiment design. |
| `selected_window_surface_diagnostics.ipynb` | Surface-deterioration diagnostics and the two ATM plots used in the thesis. |
| `predictive_atm_iv_gap_deterioration_analysis.ipynb` | Focused ATM implied-volatility gap analysis behind deterioration episodes. |
| `quote_screening_signal_validation.ipynb` | Compact quote-level signal validation diagnostic. |

## Analysis Scope

The notebook set is kept deliberately narrow so each empirical result has a
single primary location.

| Topic | Where it belongs |
|---|---|
| Headline paired predictive score | `predictive_likelihood.ipynb` |
| Four-window predictive and pricing comparison | `nonoverlapping_window_analysis.ipynb` |
| Predictive intervals and moneyness calibration | `predictive_uncertainty_calibration.ipynb` |
| ESS, resampling, tempering, and runtime | `particle_stability_numerical_behaviour.ipynb` |
| Kernel-lift numerical approximation | `rough_kernel_lift_diagnostics.ipynb` |
| Robustness checks | `robustness_q160_eval300_analysis.ipynb` |
| Surface deterioration and ATM plots | `selected_window_surface_diagnostics.ipynb` |
| Focused ATM IV gap mechanism | `predictive_atm_iv_gap_deterioration_analysis.ipynb` |
| Quote-screening signal validation | `quote_screening_signal_validation.ipynb` |

Standalone exploratory RMSE, state-path, H-robustness, pricing-grid, and broad
model-comparison notebooks have been consolidated into the analysis notebooks
above.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data and Outputs

The raw Deribit option snapshot file is expected locally at:

```text
data/raw/btc_options_snapshots_5d.csv
```

The analysis notebooks use model outputs from folders such as:

```text
outputs/common_eval_q160_eval300/
outputs/nonoverlap_q160_eval300/
outputs/robustness_q160_eval300_expanded/
outputs/comparisons/
```

The notebooks also write small derived tables and figures under
`outputs/comparisons/`.

## Running a Small Model Test

Run commands from the repository root.

Markovian SABR benchmark:

```bash
python src/normal_sabr_pf.py --raw-csv data/raw/btc_options_snapshots_5d.csv --output-dir outputs/markovian_sabr_test --start-index 0 --n-timestamps 100 --n-particles 100 --n-mc-paths 128 --n-eval-mc-paths 256 --max-options-per-timestamp 30 --beta 0.7 --log-A-process-sd 0.0 --likelihood-components price --price-likelihood-mode bidask-interval --quote-weighting equal-expiry --price-unit btc --random-seed 123
```

Memory-Augmented Rough-SABR:

```bash
python src/rough_sabr_pf.py --raw-csv data/raw/btc_options_snapshots_5d.csv --output-dir outputs/rough_sabr_test --start-index 0 --n-timestamps 100 --n-particles 100 --n-mc-paths 128 --n-eval-mc-paths 256 --max-options-per-timestamp 30 --beta 0.7 --H 0.20 --bw-n-factors 8 --log-U-init-sd 0.0 --log-U-process-sd 0.0 --likelihood-components price --price-likelihood-mode bidask-interval --quote-weighting equal-expiry --price-unit btc --random-seed 123
```

These are test-scale commands. The thesis experiments use larger paired runs
and multiple random seeds.

## Reproducibility Notes

Particle filtering and nested Monte Carlo pricing are stochastic. For model
comparison, use paired Markovian and Rough-SABR runs with the same:

| Setting | Why it matters |
|---|---|
| timestamps and quote panels | predictive scores must refer to the same observed panels |
| random seed pairing | reduces Monte Carlo noise in Markovian versus Rough-SABR differences |
| likelihood settings | keeps score differences interpretable |
| particle and evaluation Monte Carlo settings | avoids comparing numerical precision rather than model behaviour |
| price unit and quote weighting | keeps the likelihood scale consistent |

The thesis comparisons mainly use paired seeds across the selected windows. The
Monte Carlo grid diagnostic is treated separately.
