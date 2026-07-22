# Memory-Augmented Rough-SABR Particle Filter

This repository contains the computational framework developed for an MSc thesis investigating the sequential filtering of Bitcoin option-price surfaces.

The study compares two nonlinear state-space models:

1. a standard **Normal SABR particle filter**; and
2. a **Memory-Augmented Rough-SABR (MARS) particle filter**.

Both models are estimated from panels of Deribit Bitcoin option quotes using sequential Monte Carlo methods. Their empirical performance is evaluated through one-step-ahead predictive likelihood, filtered pricing accuracy, bid–ask coverage, effective sample size, and computational cost.

## Research objective

Let

```math
Y_t
=
\left\{
K_{t,j},
T_{t,j},
P^{\mathrm{bid}}_{t,j},
P^{\mathrm{ask}}_{t,j},
P^{\mathrm{mark}}_{t,j}
\right\}_{j=1}^{m_t}.
```

This denotes the panel of option quotes observed at timestamp $t$, where $K_{t,j}$ is the strike, $T_{t,j}$ is the remaining time to expiry, and $m_t$ is the number of selected option contracts.

The particle filter recursively approximates the filtering distribution

```math
p\left(X_t \mid Y_{1:t}\right),
```

where $X_t$ contains the latent forward, volatility state, model parameters, and, in the rough specification, the additional memory factors.

The principal objective is to determine whether explicitly carrying volatility memory improves the model's ability to describe and predict the evolution of the Bitcoin option surface.

## Model specifications

### Normal SABR particle filter

The Normal SABR model uses the latent state

```math
X_t^{\mathrm{N}}
=
\left(
F_t,
A_t,
\rho_t,
\nu_t
\right),
```

where:

- $F_t$ is the latent forward;
- $A_t$ is the SABR volatility factor;
- $\rho_t$ is the correlation parameter; and
- $\nu_t$ is the volatility-of-volatility parameter.

The continuous-time SABR dynamics are

```math
dF_t
=
A_t(F_t+s)^\beta\,dW_t^{(1)},
```

```math
dA_t
=
\nu_t A_t\,dW_t^{(2)},
```

with

```math
d\left\langle W^{(1)},W^{(2)}\right\rangle_t
=
\rho_t\,dt.
```

The elasticity parameter $\beta$ is fixed by default, while $\rho_t$ and $\nu_t$ evolve through slowly varying market-time transition distributions.

### Memory-Augmented Rough-SABR particle filter

The MARS model augments the particle state with a finite-dimensional representation of rough-volatility memory:

```math
X_t^{\mathrm{R}}
=
\left(
F_t,
U_t,
\rho_t,
\nu_t,
R_{1,t},
\ldots,
R_{L,t}
\right).
```

The memory variables $R_{\ell,t}$ provide a Markovian approximation to the non-Markovian rough-volatility kernel.

The rough driver is approximated by

```math
B_t
=
\frac{1}{\Gamma\left(H+\frac{1}{2}\right)}
\sum_{\ell=1}^{L}
w_\ell R_{\ell,t},
```

where $H\in(0,0.5)$ is the Hurst parameter and $(w_\ell)_{\ell=1}^{L}$ are the kernel-approximation weights.

The effective volatility factor is

```math
A_t
=
U_t
\exp\left(
\nu_t B_t
-
\frac{1}{2}\nu_t^2 v_t
\right),
```

where $U_t$ is the volatility-level process and $v_t$ is the variance correction associated with the approximated rough driver.

The forward then evolves according to

```math
dF_t
=
A_t(F_t+s)^\beta\,dW_t.
```

This construction allows every particle to carry information about the historical volatility path while retaining a finite-dimensional state suitable for sequential Monte Carlo estimation.

## Repository structure

```text
.
├── notebooks/
│   └── predictive_likelihood_analysis.ipynb
├── src/
│   ├── normal_sabr_pf.py
│   └── rough_sabr_pf.py
├── .gitignore
├── README.md
└── requirements.txt
```

The following local directories are used when running the empirical analysis:

```text
data/raw/       Raw Deribit option data
outputs/        Generated model results
```

The raw market data and generated output files are not stored directly in the repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/MarianaMarquesdeAlmeida/Memory-Augmented-Rough-SABR-MARS-Filter.git
cd Memory-Augmented-Rough-SABR-MARS-Filter
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data requirements

Place the raw Deribit option snapshot file in:

```text
data/raw/btc_options_snapshots_5d.csv
```

The model automatically identifies the relevant columns when standard names are used. The principal variables required are:

- observation timestamp;
- option strike;
- call or put indicator;
- time to expiry;
- observed forward or underlying-price proxy;
- mark option price;
- bid and ask prices;
- mark implied volatility;
- expiry identifier; and
- volume and open interest, when available.

Before filtering, the scripts remove invalid observations and restrict the sample according to maturity and log-moneyness limits.

Log-moneyness is defined as

```math
k_{t,j}
=
\log\left(
\frac{K_{t,j}}{F_t^{\mathrm{obs}}}
\right).
```

## Running the models

All commands should be executed from the repository root.

The examples below use $100$ timestamps, $100$ particles, $128$ Monte Carlo pricing paths, and a maximum of $30$ option quotes per timestamp. These settings are suitable for an initial test run. Larger experiments require substantially more computation.

### Normal SABR particle filter

```bash
python src/normal_sabr_pf.py --raw-csv data/raw/btc_options_snapshots_5d.csv --output-dir outputs/normal_sabr --start-index 0 --n-timestamps 100 --n-particles 100 --n-mc-paths 128 --n-eval-mc-paths 256 --max-options-per-timestamp 30 --beta 0.7 --log-A-process-sd 0.0 --likelihood-components price --price-likelihood-mode bidask-interval --quote-weighting equal-expiry --price-unit btc --random-seed 123
```

This command estimates the Normal SABR particle filter using option prices expressed in BTC premium units.

The additional independent random walk in $\log A_t$ is disabled through

```text
--log-A-process-sd 0.0
```

so that volatility-factor variation is generated by the SABR transition itself rather than by an additional volatility-level process.

### Memory-Augmented Rough-SABR particle filter

```bash
python src/rough_sabr_pf.py --raw-csv data/raw/btc_options_snapshots_5d.csv --output-dir outputs/rough_sabr --start-index 0 --n-timestamps 100 --n-particles 100 --n-mc-paths 128 --n-eval-mc-paths 256 --max-options-per-timestamp 30 --beta 0.7 --H 0.20 --bw-n-factors 8 --log-U-init-sd 0.0 --log-U-process-sd 0.0 --likelihood-components price --price-likelihood-mode bidask-interval --quote-weighting equal-expiry --price-unit btc --random-seed 123
```

The roughness parameter is fixed at

```math
H=0.20,
```

and the rough kernel is approximated using $L=8$ memory factors.

The arguments

```text
--log-U-init-sd 0.0
--log-U-process-sd 0.0
```

remove additional initial dispersion and process noise from the volatility-level variable $U_t$ for the controlled model comparison used in the thesis experiments.

## Important command-line arguments

| Argument | Description |
|---|---|
| `--raw-csv` | Path to the raw Deribit option dataset |
| `--output-dir` | Directory in which model outputs are stored |
| `--start-index` | Zero-based index of the first timestamp |
| `--n-timestamps` | Number of consecutive timestamps to process |
| `--n-particles` | Number of particles used in the filter |
| `--n-mc-paths` | Monte Carlo paths used in the particle likelihood |
| `--n-eval-mc-paths` | Monte Carlo paths used to evaluate filtered fit |
| `--max-options-per-timestamp` | Maximum number of option quotes retained at each timestamp |
| `--beta` | Fixed SABR elasticity parameter $\beta$ |
| `--H` | Fixed roughness parameter $H$ in the MARS model |
| `--bw-n-factors` | Number of memory factors used in the rough-kernel approximation |
| `--log-A-process-sd` | Additional market-time process noise in $\log A_t$ |
| `--log-U-process-sd` | Additional market-time process noise in $\log U_t$ |
| `--price-likelihood-mode` | Price likelihood based on the bid–ask interval or mark price |
| `--quote-weighting` | Equal-expiry or equal-quote likelihood weighting |
| `--price-unit` | Option-price unit, either BTC or USD |
| `--random-seed` | Random seed used for reproducibility |

The complete list of arguments can be displayed using:

```bash
python src/normal_sabr_pf.py --help
```

or

```bash
python src/rough_sabr_pf.py --help
```

## Observation likelihood

By default, the models use the bid–ask interval likelihood.

For a model price $\widehat{P}_{t,j}$, the pricing residual is defined as

```math
r_{t,j}
=
\begin{cases}
P^{\mathrm{bid}}_{t,j}-\widehat{P}_{t,j},
&
\widehat{P}_{t,j}<P^{\mathrm{bid}}_{t,j},
\\[4pt]
0,
&
P^{\mathrm{bid}}_{t,j}
\leq
\widehat{P}_{t,j}
\leq
P^{\mathrm{ask}}_{t,j},
\\[4pt]
\widehat{P}_{t,j}-P^{\mathrm{ask}}_{t,j},
&
\widehat{P}_{t,j}>P^{\mathrm{ask}}_{t,j}.
\end{cases}
```

A prediction therefore receives no pricing penalty when it lies inside the observed bid–ask interval.

The option-pricing contribution to the particle log likelihood is proportional to

```math
\ell_t^{(i)}
=
-\frac{1}{2}
\sum_{j=1}^{m_t}
\omega_{t,j}
\left(
\frac{r_{t,j}^{(i)}}{s_{t,j}}
\right)^2,
```

where $s_{t,j}$ is the quote-specific observation-noise scale and $\omega_{t,j}$ is the likelihood weight.

Under equal-expiry weighting,

```math
\omega_{t,j}
=
\frac{1}{n_{t,e(j)}},
```

where $n_{t,e(j)}$ is the number of selected quotes belonging to the expiry of contract $j$. Each expiry therefore contributes approximately equal total weight to the likelihood.

## Particle-filter outputs

Each model creates a separate output directory containing the principal estimation and diagnostic files.

| Output file | Description |
|---|---|
| `filtered_state_path.csv` | Filtered posterior means and dispersion measures for the latent state |
| `predictive_option_prices.csv` | One-step-ahead predictive option-price distributions |
| `loglikelihood.csv` | Timestamp-level predictive log-likelihood results |
| `surface_rmse_over_time.csv` | Filtered pricing errors across timestamps |
| `filtered_mean_option_fit.csv` | Quote-level fit from the filtered mean state |
| `ess.csv` | Effective sample size and resampling diagnostics |
| `runtime_by_timestamp.csv` | Computational time by filtering stage and timestamp |
| `model_comparison_summary.csv` | Aggregate likelihood, fit, ESS, and runtime statistics |
| `selected_observations.csv` | Option quotes selected for the filtering experiment |
| `plots/` | Automatically generated state and diagnostic plots |

The effective sample size is calculated as

```math
\mathrm{ESS}_t
=
\frac{1}
{\sum_{i=1}^{N}\left(w_t^{(i)}\right)^2},
```

where $N$ is the number of particles and $w_t^{(i)}$ is the normalized posterior weight of particle $i$.

## Predictive model comparison

The primary model-comparison statistic is the one-step-ahead predictive log likelihood

```math
\log p\left(Y_t\mid Y_{1:t-1}\right)
=
\log\left[
\sum_{i=1}^{N}
w_{t-1}^{(i)}
p\left(
Y_t\mid X_t^{(i)}
\right)
\right].
```

The cumulative predictive log likelihood is

```math
\mathcal{L}_{1:T}
=
\sum_{t=1}^{T}
\log p\left(Y_t\mid Y_{1:t-1}\right).
```

A model with a higher cumulative predictive log likelihood assigns greater probability to the subsequently observed option panels.

Predictive likelihood values should only be compared across runs using identical:

- selected timestamps;
- selected option quotes;
- likelihood components;
- quote weights;
- observation-noise scales;
- Monte Carlo settings; and
- pricing units.

The implementation omits Gaussian normalization constants from the reported likelihood. The reported values are therefore intended for controlled relative model comparison rather than as absolute likelihood measurements.

## Reproducibility

Particle filtering and nested Monte Carlo option pricing are stochastic procedures. Results may vary across random seeds because of:

- particle initialization;
- state-transition draws;
- Monte Carlo pricing draws; and
- resampling decisions.

For this reason, the empirical analysis evaluates each model across multiple random seeds rather than relying on a single run.

The same seed should be used for the Normal SABR and MARS filters when constructing paired model comparisons.

## Computational considerations

Both implementations use nested Monte Carlo simulation: each particle generates simulated terminal forward values for the option contracts included in the observation panel.

The approximate computational burden increases with

```math
N_{\mathrm{timestamps}}
\times
N_{\mathrm{particles}}
\times
N_{\mathrm{MC\ paths}}
\times
N_{\mathrm{options}}.
```

The example commands are intended as test configurations. Full empirical experiments should be run using larger particle and Monte Carlo samples only after verifying that the data paths and model outputs are correct.
