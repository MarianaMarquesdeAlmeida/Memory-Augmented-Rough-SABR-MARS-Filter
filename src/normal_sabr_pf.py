r"""
Joint SABR factor-parameter particle filter using raw Deribit option snapshots.

Purpose
-------
This script drops rough-SABR and builds one state-space model in which each
particle carries both:

    1. SABR factors under Hagan-style SABR dynamics:
           F_t  = latent/current forward factor
           A_t  = latent/current SABR volatility factor

    2. SABR parameters:
           rho_t = correlation
           nu_t  = volatility of volatility
           beta  = fixed by default, optionally filtered

The prediction step propagates F_t and A_t with the SABR SDE over market-clock
snapshot time. Optionally, an additional independent random walk can be added
to log A_t through --log-A-process-sd. This is disabled by default and is used
for the controlled comparison with the rough-SABR log-U process noise. The
update step prices the observed option quotes by Monte Carlo
under the same SABR dynamics from the current particle state to each option
expiry, and evaluates a price/IV likelihood against raw market observables.

This is a proper nested Monte Carlo particle filter. It is expensive. Start with
small values such as n_particles=200 and n_mc_paths=256 before scaling up.

Example Windows run:

    python scripts/sabr_joint_factor_parameter_pf.py ^
      --raw-csv "C:\Users\maria\Project\Thesis_project\btc_options_data_capture\data\raw\btc_options_snapshots_5d.csv" ^
      --output-dir "C:\Users\maria\Project\Thesis_project\btc_options_data_capture\data\sabr_joint_pf" ^
      --n-timestamps 100 ^
      --n-particles 300 ^
      --n-mc-paths 512 ^
      --max-options-per-timestamp 80 ^
      --likelihood-components price ^
      --price-likelihood-mode bidask-interval ^
      --price-unit btc

Notes
-----
- Deribit BTC option prices are usually quoted in BTC premium units. With
  --price-unit btc, model USD forward payoff prices are divided by the current
  particle forward F_t to compare to BTC premium quotes.
- If you use --likelihood-components both, the model IV is obtained by inverting
  the Monte Carlo model price through Black-76. This is costly and noisy.
- A full production version should replace the nested MC pricing map by a cached
  MC grid, PDE map, Hagan asymptotic formula, or neural-network pricing surrogate.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for Black implied-vol inversion in this script.") from exc

YEAR_SECONDS = 365.0 * 24.0 * 60.0 * 60.0
EPS = 1e-12


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def safe_filename(text: object) -> str:
    text = str(text)
    text = text.replace(":", "-").replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", text)
    return text.strip("_")


def pick_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {list(candidates)}")
    return None


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=float)
    m = np.nanmax(log_weights)
    if not np.isfinite(m):
        raise FloatingPointError("All log weights are non-finite.")
    w = np.exp(log_weights - m)
    s = np.nansum(w)
    if (not np.isfinite(s)) or s <= 0.0:
        raise FloatingPointError("Particle weights collapsed.")
    return w / s


def effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights * weights))


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x))


def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    m = weighted_mean(x, w)
    return float(np.sqrt(np.sum(w * (x - m) ** 2)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Weighted quantiles for one-dimensional finite arrays."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(valid):
        return np.full(len(quantiles), np.nan)

    values = values[valid]
    weights = weights[valid]
    total = weights.sum()
    if total <= 0.0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / total
    cdf = np.cumsum(weights)
    return np.interp(quantiles, cdf, values)


def logit(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    x = np.clip(x, 1e-8, 1 - 1e-8)
    return np.log(x / (1 - x))


def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def beta_to_raw(beta: float, beta_min: float, beta_max: float) -> float:
    u = (beta - beta_min) / (beta_max - beta_min)
    return float(logit(u))


def raw_to_beta(raw: np.ndarray, beta_min: float, beta_max: float) -> np.ndarray:
    return beta_min + (beta_max - beta_min) * inv_logit(raw)


def parse_option_side(values: pd.Series) -> np.ndarray:
    s = values.astype(str).str.lower().str.strip()
    out = np.where(
        s.str.startswith("c") | s.str.contains("call", regex=False),
        "call",
        np.where(s.str.startswith("p") | s.str.contains("put", regex=False), "put", "unknown"),
    )
    return out.astype(object)


def auto_scale_iv(series: pd.Series) -> pd.Series:
    """Deribit IV columns may be decimals or percentages. Convert to decimals."""
    x = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else np.nan
    if np.isfinite(med) and med > 5.0:
        return x / 100.0
    return x


# -----------------------------------------------------------------------------
# Raw observation construction
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnMap:
    time: str
    strike: str
    option_type: str
    forward: str
    maturity_years: str
    market_price: str
    bid_price: Optional[str]
    ask_price: Optional[str]
    market_iv: Optional[str]
    bid_iv: Optional[str]
    ask_iv: Optional[str]
    expiry: Optional[str]
    volume: Optional[str]
    open_interest: Optional[str]


@dataclass(frozen=True)
class ObservationSet:
    time_value: str
    t_index: int
    F_obs: float
    options: pd.DataFrame


def infer_columns(raw_df: pd.DataFrame, args: argparse.Namespace) -> ColumnMap:
    time_col = pick_column(raw_df, ["capture_time_utc", "timestamp", "time", "datetime"])
    strike_col = pick_column(raw_df, ["strike", "strike_price"])
    option_type_col = pick_column(raw_df, ["option_type", "type", "call_put", "cp"])

    if args.forward_col != "auto":
        forward_col = args.forward_col
        if forward_col not in raw_df.columns:
            raise ValueError(f"--forward-col {forward_col} not found in raw CSV.")
    else:
        forward_col = pick_column(
            raw_df,
            [
                "forward_median",
                "underlying_price",
                "index_price",
                "estimated_delivery_price",
                "underlying_index_price",
            ],
        )

    if args.maturity_col != "auto":
        maturity_col = args.maturity_col
        if maturity_col not in raw_df.columns:
            raise ValueError(f"--maturity-col {maturity_col} not found in raw CSV.")
    else:
        maturity_col = pick_column(raw_df, ["T_median", "time_to_expiry_years", "t_years", "T", "maturity_years"])

    if args.market_price_col != "auto":
        market_price_col = args.market_price_col
        if market_price_col not in raw_df.columns:
            raise ValueError(f"--market-price-col {market_price_col} not found in raw CSV.")
    else:
        market_price_col = pick_column(raw_df, ["mark_price", "mid_price", "price", "last_price"])

    bid_col = pick_column(raw_df, ["bid_price", "best_bid_price", "bid"], required=False)
    ask_col = pick_column(raw_df, ["ask_price", "best_ask_price", "ask"], required=False)
    market_iv_col = pick_column(raw_df, ["mark_iv", "chosen_iv", "mid_iv", "iv"], required=False)
    bid_iv_col = pick_column(raw_df, ["bid_iv"], required=False)
    ask_iv_col = pick_column(raw_df, ["ask_iv"], required=False)
    expiry_col = pick_column(raw_df, ["expiry", "expiration", "expiry_date"], required=False)
    volume_col = pick_column(raw_df, ["volume", "volume_usd", "stats_volume"], required=False)
    oi_col = pick_column(raw_df, ["open_interest", "oi"], required=False)

    return ColumnMap(
        time=time_col,
        strike=strike_col,
        option_type=option_type_col,
        forward=forward_col,
        maturity_years=maturity_col,
        market_price=market_price_col,
        bid_price=bid_col,
        ask_price=ask_col,
        market_iv=market_iv_col,
        bid_iv=bid_iv_col,
        ask_iv=ask_iv_col,
        expiry=expiry_col,
        volume=volume_col,
        open_interest=oi_col,
    )


def prepare_raw_options(raw_df: pd.DataFrame, cols: ColumnMap, args: argparse.Namespace) -> pd.DataFrame:
    df = raw_df.copy()
    df[cols.time] = pd.to_datetime(df[cols.time], utc=True, errors="coerce")
    df["capture_time_utc"] = df[cols.time].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["capture_time_dt"] = df[cols.time]

    df["strike"] = pd.to_numeric(df[cols.strike], errors="coerce")
    df["F_obs"] = pd.to_numeric(df[cols.forward], errors="coerce")
    df["T_years"] = pd.to_numeric(df[cols.maturity_years], errors="coerce")
    df["market_price"] = pd.to_numeric(df[cols.market_price], errors="coerce")
    df["option_type_clean"] = parse_option_side(df[cols.option_type])

    if cols.bid_price is not None:
        df["bid_price"] = pd.to_numeric(df[cols.bid_price], errors="coerce")
    else:
        df["bid_price"] = np.nan

    if cols.ask_price is not None:
        df["ask_price"] = pd.to_numeric(df[cols.ask_price], errors="coerce")
    else:
        df["ask_price"] = np.nan

    if cols.market_iv is not None:
        df["market_iv"] = auto_scale_iv(df[cols.market_iv])
    else:
        df["market_iv"] = np.nan

    if cols.bid_iv is not None:
        df["bid_iv"] = auto_scale_iv(df[cols.bid_iv])
    else:
        df["bid_iv"] = np.nan

    if cols.ask_iv is not None:
        df["ask_iv"] = auto_scale_iv(df[cols.ask_iv])
    else:
        df["ask_iv"] = np.nan

    if cols.expiry is not None:
        df["expiry"] = df[cols.expiry].astype(str)
    else:
        # Use rounded maturity as a fallback expiry bucket.
        df["expiry"] = "T_" + (df["T_years"] * 365.0).round(4).astype(str)

    if cols.volume is not None:
        df["volume"] = pd.to_numeric(df[cols.volume], errors="coerce")
    else:
        df["volume"] = np.nan

    if cols.open_interest is not None:
        df["open_interest"] = pd.to_numeric(df[cols.open_interest], errors="coerce")
    else:
        df["open_interest"] = np.nan

    # Use only sensible rows.
    keep = (
        df["capture_time_dt"].notna()
        & np.isfinite(df["strike"])
        & np.isfinite(df["F_obs"])
        & np.isfinite(df["T_years"])
        & np.isfinite(df["market_price"])
        & (df["strike"] > 0.0)
        & (df["F_obs"] > 0.0)
        & (df["T_years"] > args.min_maturity_days / 365.0)
        & (df["T_years"] <= args.max_maturity_days / 365.0)
        & (df["market_price"] >= 0.0)
        & (df["option_type_clean"] != "unknown")
    )

    df = df[keep].copy()
    df["log_moneyness"] = np.log(df["strike"] / df["F_obs"])
    df = df[np.abs(df["log_moneyness"]) <= args.max_abs_log_moneyness].copy()
    df = df.sort_values(["capture_time_dt", "T_years", "strike"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid option rows remain after cleaning.")

    return df


def selected_times(df: pd.DataFrame, start_index: int, n_timestamps: int) -> list[str]:
    unique_times = (
        df.sort_values("capture_time_dt")["capture_time_utc"]
        .drop_duplicates()
        .tolist()
    )

    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if start_index >= len(unique_times):
        raise ValueError(
            f"--start-index {start_index} is outside the available range "
            f"0 to {len(unique_times) - 1}."
        )

    times = unique_times[start_index : start_index + n_timestamps]
    if not times:
        raise ValueError("No timestamps selected.")
    return times


def subsample_options(g: pd.DataFrame, max_options: int, rng: np.random.Generator) -> pd.DataFrame:
    if max_options <= 0 or len(g) <= max_options:
        return g.copy()

    # Deterministic-ish surface coverage: split by expiry and take evenly across moneyness.
    groups = list(g.groupby("expiry", sort=True))
    per_group = max(1, max_options // max(1, len(groups)))
    chunks = []
    for _, h in groups:
        h = h.sort_values("log_moneyness")
        if len(h) <= per_group:
            chunks.append(h)
        else:
            idx = np.linspace(0, len(h) - 1, per_group).round().astype(int)
            chunks.append(h.iloc[np.unique(idx)])

    out = pd.concat(chunks, axis=0)

    # If still too many, take evenly across maturity/moneyness ordering.
    if len(out) > max_options:
        out = out.sort_values(["T_years", "log_moneyness"])
        idx = np.linspace(0, len(out) - 1, max_options).round().astype(int)
        out = out.iloc[np.unique(idx)]

    return out.sort_values(["T_years", "strike"]).reset_index(drop=True)


def build_observation_sets(
    df: pd.DataFrame,
    times: list[str],
    start_index: int,
    args: argparse.Namespace,
) -> list[ObservationSet]:
    obs_sets: list[ObservationSet] = []
    rng = np.random.default_rng(args.random_seed + 17)
    grouped = {t: g for t, g in df[df["capture_time_utc"].isin(times)].groupby("capture_time_utc", sort=False)}

    for t_idx, t in enumerate(times, start=start_index):
        g = grouped.get(t)
        if g is None or g.empty:
            continue
        g = subsample_options(g, args.max_options_per_timestamp, rng)
        F_obs = float(np.nanmedian(g["F_obs"]))
        obs_sets.append(ObservationSet(time_value=t, t_index=t_idx, F_obs=F_obs, options=g))

    if not obs_sets:
        raise ValueError("Could not build any observation sets.")
    return obs_sets


def estimate_initial_atm_iv(first_obs: ObservationSet) -> float:
    g = first_obs.options.copy()
    g = g[np.isfinite(g["market_iv"]) & (g["market_iv"] > 0)]
    if g.empty:
        return 0.70  # fallback for BTC options
    g["abs_m"] = np.abs(g["log_moneyness"])
    return float(g.sort_values("abs_m").head(max(3, min(10, len(g))))["market_iv"].median())


# -----------------------------------------------------------------------------
# State transform and prediction
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StateConfig:
    beta_fixed: float
    estimate_beta: bool
    beta_min: float
    beta_max: float
    shift: float
    F_floor: float
    A_floor: float
    nu_floor: float
    nu_cap: float


def state_dim(cfg: StateConfig) -> int:
    return 5 if cfg.estimate_beta else 4


def pack_state(F: np.ndarray, A: np.ndarray, rho: np.ndarray, nu: np.ndarray, beta: Optional[np.ndarray], cfg: StateConfig) -> np.ndarray:
    if cfg.estimate_beta:
        if beta is None:
            raise ValueError("beta must be supplied when estimate_beta=True")
        beta_raw = beta_to_raw(float(cfg.beta_fixed), cfg.beta_min, cfg.beta_max) if np.ndim(beta) == 0 else logit((beta - cfg.beta_min) / (cfg.beta_max - cfg.beta_min))
        return np.column_stack([
            np.log(np.maximum(F, cfg.F_floor)),
            np.log(np.maximum(A, cfg.A_floor)),
            np.arctanh(np.clip(rho, -0.999, 0.999)),
            np.log(np.maximum(nu, cfg.nu_floor)),
            beta_raw,
        ])
    return np.column_stack([
        np.log(np.maximum(F, cfg.F_floor)),
        np.log(np.maximum(A, cfg.A_floor)),
        np.arctanh(np.clip(rho, -0.999, 0.999)),
        np.log(np.maximum(nu, cfg.nu_floor)),
    ])


def unpack_state(particles: np.ndarray, cfg: StateConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F = np.maximum(np.exp(particles[:, 0]), cfg.F_floor)
    A = np.maximum(np.exp(particles[:, 1]), cfg.A_floor)
    rho = np.tanh(particles[:, 2])
    nu = np.clip(np.exp(particles[:, 3]), cfg.nu_floor, cfg.nu_cap)
    if cfg.estimate_beta:
        beta = raw_to_beta(particles[:, 4], cfg.beta_min, cfg.beta_max)
    else:
        beta = np.full(len(particles), cfg.beta_fixed, dtype=float)
    return F, A, rho, nu, beta


def initialize_particles(first_obs: ObservationSet, n_particles: int, cfg: StateConfig, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.random_seed)
    F0 = first_obs.F_obs
    atm_iv = estimate_initial_atm_iv(first_obs)

    # For SABR dF = A F^beta dW, the local log-vol scale is roughly A F^(beta-1).
    # Hence A0 ~= ATM_IV * F^(1-beta).
    A0 = max(atm_iv * (F0 + cfg.shift) ** (1.0 - cfg.beta_fixed), cfg.A_floor)
    rho0 = args.rho0
    nu0 = args.nu0
    beta0 = cfg.beta_fixed

    base = np.array([
        math.log(F0),
        math.log(A0),
        math.atanh(np.clip(rho0, -0.999, 0.999)),
        math.log(nu0),
    ], dtype=float)
    init_sd = np.array([
        args.log_F_init_sd,
        args.log_A_init_sd,
        args.rho_raw_init_sd,
        args.log_nu_init_sd,
    ], dtype=float)

    if cfg.estimate_beta:
        base = np.r_[base, beta_to_raw(beta0, cfg.beta_min, cfg.beta_max)]
        init_sd = np.r_[init_sd, args.beta_raw_init_sd]

    return base[None, :] + rng.normal(scale=init_sd[None, :], size=(n_particles, len(base)))


def predict_particles_sabr(
    particles: np.ndarray,
    dt_years: float,
    rng: np.random.Generator,
    cfg: StateConfig,
    args: argparse.Namespace,
) -> np.ndarray:
    if dt_years <= 0.0:
        return particles.copy()

    F, A, rho, nu, beta = unpack_state(particles, cfg)
    n = len(F)
    sqrt_dt = math.sqrt(dt_years)

    z1 = rng.normal(size=n)
    zind = rng.normal(size=n)
    z2 = rho * z1 + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * zind

    C = np.maximum(F + cfg.shift, cfg.F_floor) ** beta
    F_new = F + A * C * sqrt_dt * z1
    F_new = np.maximum(F_new, cfg.F_floor)

    # Standard SABR volatility-factor transition.
    A_new = A * np.exp(-0.5 * nu * nu * dt_years + nu * sqrt_dt * z2)

    # Optional additional market-time random walk in log A.  This is separate
    # from the SABR SDE and is included only to match the extra log-U process
    # flexibility used in the rough-SABR filter.  At the reference interval,
    # its log standard deviation is args.log_A_process_sd.
    scale = max(sqrt_dt / math.sqrt(args.reference_dt_years), 0.05)
    if args.log_A_process_sd > 0.0:
        log_A_rw_shock = rng.normal(
            scale=args.log_A_process_sd * scale,
            size=n,
        )
        A_new *= np.exp(log_A_rw_shock)
    A_new = np.maximum(A_new, cfg.A_floor)

    out = particles.copy()
    out[:, 0] = np.log(F_new)
    out[:, 1] = np.log(A_new)

    # Parameter evolution: this is not Hagan's option-pricing SDE; it is the
    # market-time drift/noise prior for slowly changing parameters.
    out[:, 2] += rng.normal(scale=args.rho_raw_process_sd * scale, size=n)
    out[:, 3] += rng.normal(scale=args.log_nu_process_sd * scale, size=n)
    if cfg.estimate_beta:
        out[:, 4] += rng.normal(scale=args.beta_raw_process_sd * scale, size=n)

    return out


# -----------------------------------------------------------------------------
# SABR Monte Carlo pricing
# -----------------------------------------------------------------------------

def simulate_terminal_forward_sabr(
    F0: np.ndarray,
    A0: np.ndarray,
    rho: np.ndarray,
    nu: np.ndarray,
    beta: np.ndarray,
    T: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    cfg: StateConfig,
    antithetic: bool,
) -> np.ndarray:
    """Simulate terminal forward F_T for every particle. Shape: (n_particles, n_paths)."""
    n_particles = len(F0)
    n_steps = max(1, int(n_steps))
    dt = max(float(T) / n_steps, 1e-12)
    sqrt_dt = math.sqrt(dt)

    if antithetic:
        half = max(1, n_paths // 2)
        n_eff = 2 * half
    else:
        half = n_paths
        n_eff = n_paths

    F = np.repeat(F0[:, None], n_eff, axis=1)
    A = np.repeat(A0[:, None], n_eff, axis=1)

    rho2 = rho[:, None]
    nu2 = nu[:, None]
    beta2 = beta[:, None]
    corr_scale = np.sqrt(np.maximum(1.0 - rho2 * rho2, 0.0))

    for _ in range(n_steps):
        z1_half = rng.normal(size=(n_particles, half))
        zi_half = rng.normal(size=(n_particles, half))
        if antithetic:
            z1 = np.concatenate([z1_half, -z1_half], axis=1)
            zi = np.concatenate([zi_half, -zi_half], axis=1)
        else:
            z1 = z1_half
            zi = zi_half

        z2 = rho2 * z1 + corr_scale * zi
        C = np.maximum(F + cfg.shift, cfg.F_floor) ** beta2
        F = np.maximum(F + A * C * sqrt_dt * z1, cfg.F_floor)
        A = np.maximum(A * np.exp(-0.5 * nu2 * nu2 * dt + nu2 * sqrt_dt * z2), cfg.A_floor)

    return F


def price_options_from_terminal(
    F_terminal: np.ndarray,
    F0_particles: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    price_unit: str,
    chunk_size: int = 32,
) -> np.ndarray:
    n_particles = F_terminal.shape[0]
    n_options = len(strikes)
    out = np.empty((n_particles, n_options), dtype=float)

    for start in range(0, n_options, chunk_size):
        end = min(start + chunk_size, n_options)
        K = strikes[start:end][None, None, :]
        cp = option_types[start:end]
        Ft = F_terminal[:, :, None]

        call_payoff = np.maximum(Ft - K, 0.0)
        put_payoff = np.maximum(K - Ft, 0.0)
        payoff = np.where(cp[None, None, :] == "call", call_payoff, put_payoff)
        price_usd = np.mean(payoff, axis=1)

        if price_unit == "btc":
            out[:, start:end] = price_usd / np.maximum(F0_particles[:, None], EPS)
        elif price_unit == "usd":
            out[:, start:end] = price_usd
        else:
            raise ValueError("price_unit must be 'btc' or 'usd'.")

    return out


def mc_price_observation_set(
    particles: np.ndarray,
    obs: ObservationSet,
    cfg: StateConfig,
    args: argparse.Namespace,
    rng: np.random.Generator,
    n_mc_paths: Optional[int] = None,
) -> tuple[np.ndarray, list[int]]:
    """Return model prices of shape (n_particles, n_options)."""
    if n_mc_paths is None:
        n_mc_paths = args.n_mc_paths

    F0, A0, rho, nu, beta = unpack_state(particles, cfg)
    g = obs.options.reset_index(drop=True)
    n_options = len(g)
    model_prices = np.empty((len(particles), n_options), dtype=float)

    # Group by rounded maturity to share one MC terminal simulation across strikes with same expiry.
    group_keys = g.groupby("expiry", sort=False).groups
    priced_indices: list[int] = []

    for expiry_key, idx_obj in group_keys.items():
        idx = np.array(list(idx_obj), dtype=int)
        h = g.iloc[idx]
        T = float(np.nanmedian(h["T_years"]))
        if not np.isfinite(T) or T <= 0.0:
            continue

        n_steps = max(1, int(math.ceil(T * 365.0 / args.mc_days_per_step)))
        n_steps = min(n_steps, args.max_mc_steps)

        F_T = simulate_terminal_forward_sabr(
            F0=F0,
            A0=A0,
            rho=rho,
            nu=nu,
            beta=beta,
            T=T,
            n_paths=n_mc_paths,
            n_steps=n_steps,
            rng=rng,
            cfg=cfg,
            antithetic=args.antithetic,
        )

        prices = price_options_from_terminal(
            F_terminal=F_T,
            F0_particles=F0,
            strikes=h["strike"].to_numpy(dtype=float),
            option_types=h["option_type_clean"].to_numpy(dtype=object),
            price_unit=args.price_unit,
            chunk_size=args.payoff_chunk_size,
        )
        model_prices[:, idx] = prices
        priced_indices.extend(idx.tolist())

    if len(priced_indices) < n_options:
        missing = sorted(set(range(n_options)).difference(priced_indices))
        model_prices[:, missing] = np.nan

    return model_prices, priced_indices


# -----------------------------------------------------------------------------
# Black prices and IV inversion from MC model price
# -----------------------------------------------------------------------------

def black76_price_usd(F: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, option_type: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    vol_sqrt_T = np.maximum(sigma * np.sqrt(np.maximum(T, 1e-12)), 1e-12)
    d1 = (np.log(np.maximum(F, EPS) / np.maximum(K, EPS)) + 0.5 * sigma * sigma * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return np.where(option_type == "call", call, put)


def implied_vol_from_model_price(
    model_price: np.ndarray,
    F0_particles: np.ndarray,
    strikes: np.ndarray,
    T_years: np.ndarray,
    option_types: np.ndarray,
    price_unit: str,
    max_iter: int = 50,
) -> np.ndarray:
    """Invert Black-76 price to IV. Shape model_price: (n_particles, n_options)."""
    n_particles, n_options = model_price.shape
    F = F0_particles[:, None]
    K = strikes[None, :]
    T = T_years[None, :]
    cp = option_types[None, :]

    price_usd = model_price * F if price_unit == "btc" else model_price.copy()

    lo = np.full_like(price_usd, 1e-4)
    hi = np.full_like(price_usd, 5.0)

    intrinsic = np.where(cp == "call", np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    valid = np.isfinite(price_usd) & (price_usd >= intrinsic - 1e-10)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = black76_price_usd(F, K, T, mid, cp)
        too_low = p_mid < price_usd
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)

    iv = 0.5 * (lo + hi)
    iv[~valid] = np.nan
    return iv


# -----------------------------------------------------------------------------
# Likelihood
# -----------------------------------------------------------------------------

def option_noise_scales(obs: ObservationSet, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    g = obs.options.reset_index(drop=True)
    market = g["market_price"].to_numpy(dtype=float)
    bid = g["bid_price"].to_numpy(dtype=float)
    ask = g["ask_price"].to_numpy(dtype=float)

    half_spread = 0.5 * (ask - bid)
    valid_spread = np.isfinite(half_spread) & (half_spread > 0.0) & np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)

    price_sd = np.maximum(args.price_abs_sd_floor, args.price_rel_sd_floor * np.maximum(np.abs(market), args.price_abs_sd_floor))
    price_sd = np.where(valid_spread, np.maximum(price_sd, half_spread), price_sd)

    # Mildly downweight far wings and missing bid/ask.
    m = np.abs(g["log_moneyness"].to_numpy(dtype=float))
    multiplier = np.ones(len(g), dtype=float)
    multiplier *= np.where(m > args.far_log_moneyness_cutoff, args.far_moneyness_multiplier, 1.0)
    multiplier *= np.where(valid_spread, 1.0, args.missing_bidask_multiplier)
    price_sd *= np.clip(multiplier, 1.0, args.noise_total_max_multiplier)

    iv = g["market_iv"].to_numpy(dtype=float)
    bid_iv = g["bid_iv"].to_numpy(dtype=float)
    ask_iv = g["ask_iv"].to_numpy(dtype=float)
    iv_half_spread = 0.5 * (ask_iv - bid_iv)
    valid_iv_spread = np.isfinite(iv_half_spread) & (iv_half_spread > 0.0)
    iv_sd = np.full(len(g), args.iv_obs_sd, dtype=float)
    iv_sd = np.where(valid_iv_spread, np.maximum(iv_sd, iv_half_spread), iv_sd)
    iv_sd *= np.clip(multiplier, 1.0, args.noise_total_max_multiplier)

    return price_sd, iv_sd


def log_likelihood_particles(
    particles: np.ndarray,
    obs: ObservationSet,
    cfg: StateConfig,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    model_price, priced_indices = mc_price_observation_set(particles, obs, cfg, args, rng)
    g = obs.options.reset_index(drop=True)
    n_particles = len(particles)

    market_price = g["market_price"].to_numpy(dtype=float)
    bid = g["bid_price"].to_numpy(dtype=float)
    ask = g["ask_price"].to_numpy(dtype=float)
    market_iv = g["market_iv"].to_numpy(dtype=float)
    price_sd, iv_sd = option_noise_scales(obs, args)

    expiry_counts = g.groupby("expiry")["expiry"].transform("count").to_numpy(dtype=float)
    if args.quote_weighting == "equal-expiry":
        weights = 1.0 / np.maximum(expiry_counts, 1.0)
    elif args.quote_weighting == "equal-quote":
        weights = np.ones(len(g), dtype=float)
    else:
        raise ValueError("quote_weighting must be equal-expiry or equal-quote")

    ll = np.zeros(n_particles, dtype=float)

    if args.likelihood_components in {"price", "both"}:
        valid_model = np.isfinite(model_price)
        if args.price_likelihood_mode == "bidask-interval":
            valid_band = np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)
            below = np.maximum(bid[None, :] - model_price, 0.0)
            above = np.maximum(model_price - ask[None, :], 0.0)
            dist = below + above
            # Fallback to mark residual when no valid bid/ask band exists.
            mark_resid = model_price - market_price[None, :]
            resid = np.where(valid_band[None, :], dist, mark_resid)
        elif args.price_likelihood_mode == "mark":
            resid = model_price - market_price[None, :]
        else:
            raise ValueError("Unknown price_likelihood_mode")

        std = resid / price_sd[None, :]
        std = np.where(valid_model, std, 1e6)
        ll += -0.5 * args.price_likelihood_weight * np.sum(weights[None, :] * std * std, axis=1)

    if args.likelihood_components in {"iv", "both"}:
        F0_particles, _, _, _, _ = unpack_state(particles, cfg)
        model_iv = implied_vol_from_model_price(
            model_price=model_price,
            F0_particles=F0_particles,
            strikes=g["strike"].to_numpy(dtype=float),
            T_years=g["T_years"].to_numpy(dtype=float),
            option_types=g["option_type_clean"].to_numpy(dtype=object),
            price_unit=args.price_unit,
            max_iter=args.iv_inversion_iters,
        )
        valid_iv = np.isfinite(model_iv) & np.isfinite(market_iv)[None, :] & (market_iv[None, :] > 0.0)
        resid_iv = model_iv - market_iv[None, :]
        std_iv = resid_iv / iv_sd[None, :]
        std_iv = np.where(valid_iv, std_iv, 0.0)
        valid_counts = np.sum(valid_iv, axis=1)
        penalty = np.sum(weights[None, :] * std_iv * std_iv, axis=1)
        penalty = np.where(valid_counts > 0, penalty, 1e6)
        ll += -0.5 * args.iv_likelihood_weight * penalty

    diagnostics = {
        "n_options": float(len(g)),
        "n_priced": float(len(priced_indices)),
        "mean_market_price": float(np.nanmean(market_price)),
        "median_price_sd": float(np.nanmedian(price_sd)),
    }
    return ll, diagnostics, model_price


# -----------------------------------------------------------------------------
# Tempered update
# -----------------------------------------------------------------------------

def ess_from_log_weights(log_weights: np.ndarray) -> float:
    return effective_sample_size(normalize_log_weights(log_weights))


def choose_tempering_increment(log_weights: np.ndarray, log_likelihood: np.ndarray, remaining: float, target_ess: float) -> float:
    if remaining <= 0.0:
        return 0.0
    full_ess = ess_from_log_weights(log_weights + remaining * log_likelihood)
    if full_ess >= target_ess:
        return remaining

    low = 0.0
    high = remaining
    for _ in range(35):
        mid = 0.5 * (low + high)
        ess_mid = ess_from_log_weights(log_weights + mid * log_likelihood)
        if ess_mid >= target_ess:
            low = mid
        else:
            high = mid
    return max(low, 1e-4)


def tempered_update(
    particles: np.ndarray,
    weights: np.ndarray,
    log_likelihood: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    n = len(weights)
    log_weights = np.log(weights + 1e-300)
    lambda_value = 0.0
    n_steps = 0
    n_resamples = 0
    min_ess = effective_sample_size(weights)
    target_ess = args.tempering_ess_fraction * n

    while lambda_value < 1.0 - 1e-12:
        if n_steps >= args.max_tempering_steps:
            delta = 1.0 - lambda_value
        else:
            delta = choose_tempering_increment(log_weights, log_likelihood, 1.0 - lambda_value, target_ess)
            delta = min(delta, 1.0 - lambda_value)

        log_weights = log_weights + delta * log_likelihood
        weights = normalize_log_weights(log_weights)
        ess = effective_sample_size(weights)
        min_ess = min(min_ess, ess)
        lambda_value += delta
        n_steps += 1

        if lambda_value < 1.0 - 1e-12:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            log_likelihood = log_likelihood[idx]
            weights = np.full(n, 1.0 / n)
            log_weights = np.log(weights + 1e-300)
            n_resamples += 1

    return particles, weights, {
        "min_tempered_ess": float(min_ess),
        "n_tempering_steps": int(n_steps),
        "n_tempering_resamples": int(n_resamples),
    }


# -----------------------------------------------------------------------------
# Evaluation outputs
# -----------------------------------------------------------------------------

def state_summary_row(particles: np.ndarray, weights: np.ndarray, cfg: StateConfig) -> dict[str, float]:
    F, A, rho, nu, beta = unpack_state(particles, cfg)
    row = {}
    for name, arr in [("F", F), ("A", A), ("rho", rho), ("nu", nu), ("beta", beta)]:
        row[name] = weighted_mean(arr, weights)
        row[f"{name}_std"] = weighted_std(arr, weights)
    # A convenient Black-like local log-vol scale near ATM.
    atm_logvol = A / np.maximum(F, EPS) ** (1.0 - beta)
    row["atm_logvol_proxy"] = weighted_mean(atm_logvol, weights)
    row["atm_logvol_proxy_std"] = weighted_std(atm_logvol, weights)
    return row


def predictive_price_summary(
    model_price: np.ndarray,
    particle_weights: np.ndarray,
    obs: ObservationSet,
) -> pd.DataFrame:
    """
    Summarize the pre-update predictive option-price distribution quote by quote.
    """
    g = obs.options.reset_index(drop=True).copy()
    w = np.asarray(particle_weights, dtype=float)
    w = w / np.sum(w)

    means = np.full(model_price.shape[1], np.nan)
    stds = np.full(model_price.shape[1], np.nan)
    q05 = np.full(model_price.shape[1], np.nan)
    q25 = np.full(model_price.shape[1], np.nan)
    q50 = np.full(model_price.shape[1], np.nan)
    q75 = np.full(model_price.shape[1], np.nan)
    q95 = np.full(model_price.shape[1], np.nan)

    for j in range(model_price.shape[1]):
        x = model_price[:, j]
        valid = np.isfinite(x) & np.isfinite(w)
        if not np.any(valid):
            continue

        xj = x[valid]
        wj = w[valid]
        wj = wj / np.sum(wj)

        mean_j = float(np.sum(wj * xj))
        means[j] = mean_j
        stds[j] = float(np.sqrt(np.sum(wj * (xj - mean_j) ** 2)))
        q = weighted_quantile(xj, wj, np.array([0.05, 0.25, 0.50, 0.75, 0.95]))
        q05[j], q25[j], q50[j], q75[j], q95[j] = q

    g["predictive_price_mean"] = means
    g["predictive_price_std"] = stds
    g["predictive_price_q05"] = q05
    g["predictive_price_q25"] = q25
    g["predictive_price_q50"] = q50
    g["predictive_price_q75"] = q75
    g["predictive_price_q95"] = q95
    g["predictive_price_error_vs_market"] = g["predictive_price_mean"] - g["market_price"]

    valid_band = (
        np.isfinite(g["bid_price"])
        & np.isfinite(g["ask_price"])
        & (g["ask_price"] >= g["bid_price"])
    )
    g["predictive_mean_inside_bidask"] = (
        valid_band
        & (g["predictive_price_mean"] >= g["bid_price"])
        & (g["predictive_price_mean"] <= g["ask_price"])
    )
    g["capture_time_utc"] = obs.time_value
    g["t_index"] = obs.t_index

    first_cols = ["capture_time_utc", "t_index"]
    return g[first_cols + [c for c in g.columns if c not in first_cols]]


def one_particle_from_summary(row: dict[str, float], cfg: StateConfig) -> np.ndarray:
    F = np.array([row["F"]])
    A = np.array([row["A"]])
    rho = np.array([row["rho"]])
    nu = np.array([row["nu"]])
    beta = np.array([row["beta"]])
    return pack_state(F, A, rho, nu, beta if cfg.estimate_beta else None, cfg)


def evaluate_filtered_fit(
    summary_row: dict[str, float],
    obs: ObservationSet,
    cfg: StateConfig,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, float], pd.DataFrame]:
    p = one_particle_from_summary(summary_row, cfg)
    eval_paths = args.n_eval_mc_paths if args.n_eval_mc_paths > 0 else max(args.n_mc_paths, 256)
    model_price, _ = mc_price_observation_set(p, obs, cfg, args, rng, n_mc_paths=eval_paths)
    model_price = model_price[0]

    g = obs.options.reset_index(drop=True).copy()
    g["model_price_filtered_mean"] = model_price
    g["price_residual"] = g["model_price_filtered_mean"] - g["market_price"]

    valid = np.isfinite(g["price_residual"])
    price_rmse = float(np.sqrt(np.nanmean(g.loc[valid, "price_residual"] ** 2))) if valid.any() else np.nan

    valid_band = np.isfinite(g["bid_price"]) & np.isfinite(g["ask_price"]) & (g["ask_price"] >= g["bid_price"])
    inside = valid_band & (g["model_price_filtered_mean"] >= g["bid_price"]) & (g["model_price_filtered_mean"] <= g["ask_price"])
    inside_rate = float(inside.sum() / max(valid_band.sum(), 1)) if valid_band.any() else np.nan

    out = {
        "fit_price_rmse": price_rmse,
        "fit_inside_bidask_rate": inside_rate,
        "fit_mean_abs_price_residual": float(np.nanmean(np.abs(g["price_residual"]))) if valid.any() else np.nan,
    }
    return out, g


# -----------------------------------------------------------------------------
# Filter driver
# -----------------------------------------------------------------------------

def dt_years_between(obs_sets: list[ObservationSet]) -> np.ndarray:
    times = pd.to_datetime(pd.Series([o.time_value for o in obs_sets]), utc=True, errors="coerce")
    dt_sec = times.diff().dt.total_seconds().to_numpy()
    out = np.zeros(len(obs_sets), dtype=float)
    for i in range(1, len(obs_sets)):
        if np.isfinite(dt_sec[i]) and dt_sec[i] > 0.0:
            out[i] = float(dt_sec[i] / YEAR_SECONDS)
        else:
            out[i] = 0.0
    return out


def forward_anchor_loglike(particles: np.ndarray, obs: ObservationSet, cfg: StateConfig, args: argparse.Namespace) -> np.ndarray:
    if args.forward_obs_rel_sd <= 0.0:
        return np.zeros(len(particles), dtype=float)
    F, _, _, _, _ = unpack_state(particles, cfg)
    sd = args.forward_obs_rel_sd * max(obs.F_obs, EPS)
    z = (F - obs.F_obs) / sd
    return -0.5 * z * z


def run_filter(
    obs_sets: list[ObservationSet],
    cfg: StateConfig,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(args.random_seed)
    particles = initialize_particles(obs_sets[0], args.n_particles, cfg, args)
    weights = np.full(args.n_particles, 1.0 / args.n_particles)
    dts = dt_years_between(obs_sets)

    filtered_rows = []
    ess_rows = []
    comparison_chunks = []
    loglik_rows = []
    rmse_rows = []
    runtime_rows = []
    predictive_chunks = []

    for k, obs in enumerate(obs_sets):
        timestamp_start = time.perf_counter()

        predict_seconds = 0.0
        if k > 0:
            predict_start = time.perf_counter()
            particles = predict_particles_sabr(particles, dts[k], rng, cfg, args)
            predict_seconds = time.perf_counter() - predict_start

        likelihood_start = time.perf_counter()
        mc_rng = np.random.default_rng(args.random_seed + 100_000 + k)
        ll_options, like_diag, predictive_model_price = log_likelihood_particles(
            particles, obs, cfg, args, mc_rng
        )

        predictive_chunks.append(
            predictive_price_summary(
                model_price=predictive_model_price,
                particle_weights=weights,
                obs=obs,
            )
        )

        ll_anchor = forward_anchor_loglike(particles, obs, cfg, args)
        ll_total = ll_options + ll_anchor
        likelihood_seconds = time.perf_counter() - likelihood_start

        pre_update_ess = effective_sample_size(weights)
        log_pred_like_options = particle_log_predictive_likelihood(weights, ll_options)
        log_pred_like_total = particle_log_predictive_likelihood(weights, ll_total)
        n_quotes = int(like_diag.get("n_options", len(obs.options)))

        update_start = time.perf_counter()
        if args.use_tempering:
            particles, weights, temp_diag = tempered_update(particles, weights, ll_total, rng, args)
        else:
            logw = np.log(weights + 1e-300) + ll_total
            weights = normalize_log_weights(logw)
            temp_diag = {
                "min_tempered_ess": effective_sample_size(weights),
                "n_tempering_steps": 1,
                "n_tempering_resamples": 0,
            }
        update_seconds = time.perf_counter() - update_start

        ess = effective_sample_size(weights)
        row = state_summary_row(particles, weights, cfg)
        row.update({
            "capture_time_utc": obs.time_value,
            "t_index": obs.t_index,
            "dt_years": dts[k],
            "ess": ess,
            "pre_update_ess": pre_update_ess,
            "log_predictive_likelihood": log_pred_like_options,
            "avg_log_predictive_likelihood_per_quote": log_pred_like_options / max(n_quotes, 1),
            "log_predictive_likelihood_with_forward_anchor": log_pred_like_total,
            "avg_log_predictive_likelihood_with_forward_anchor_per_quote": log_pred_like_total / max(n_quotes, 1),
            **temp_diag,
            **like_diag,
        })

        evaluation_seconds = 0.0
        if args.evaluate_filtered_fit:
            eval_start = time.perf_counter()
            eval_rng = np.random.default_rng(args.random_seed + 200_000 + k)
            fit_diag, comp = evaluate_filtered_fit(row, obs, cfg, args, eval_rng)
            evaluation_seconds = time.perf_counter() - eval_start
            row.update(fit_diag)

            rmse_row = {
                "capture_time_utc": obs.time_value,
                "t_index": obs.t_index,
                "n_quotes": n_quotes,
                **fit_diag,
            }
            rmse_rows.append(rmse_row)

            # obs.options may already contain capture_time_utc and/or t_index
            # from the raw-data preparation stage. Assign instead of insert so
            # reruns do not fail with duplicate-column errors.
            comp = comp.copy()
            comp["capture_time_utc"] = obs.time_value
            comp["t_index"] = obs.t_index
            first_cols = ["capture_time_utc", "t_index"]
            other_cols = [c for c in comp.columns if c not in first_cols]
            comp = comp[first_cols + other_cols]
            comparison_chunks.append(comp)

        timestamp_seconds = time.perf_counter() - timestamp_start

        filtered_rows.append(row)
        ess_rows.append({
            "capture_time_utc": obs.time_value,
            "t_index": obs.t_index,
            "ess": ess,
            "pre_update_ess": pre_update_ess,
            "min_tempered_ess": float(temp_diag["min_tempered_ess"]),
            "n_tempering_steps": int(temp_diag["n_tempering_steps"]),
            "n_tempering_resamples": int(temp_diag["n_tempering_resamples"]),
        })
        loglik_rows.append({
            "capture_time_utc": obs.time_value,
            "t_index": obs.t_index,
            "n_quotes": n_quotes,
            "pre_update_ess": pre_update_ess,
            "post_update_ess": ess,
            "log_predictive_likelihood": log_pred_like_options,
            "avg_log_predictive_likelihood_per_quote": log_pred_like_options / max(n_quotes, 1),
            "log_predictive_likelihood_with_forward_anchor": log_pred_like_total,
            "avg_log_predictive_likelihood_with_forward_anchor_per_quote": log_pred_like_total / max(n_quotes, 1),
            "gaussian_constants_omitted": True,
        })
        runtime_rows.append({
            "capture_time_utc": obs.time_value,
            "t_index": obs.t_index,
            "n_quotes": n_quotes,
            "predict_seconds": predict_seconds,
            "likelihood_seconds": likelihood_seconds,
            "update_seconds": update_seconds,
            "evaluation_seconds": evaluation_seconds,
            "timestamp_seconds": timestamp_seconds,
        })

        if ess < args.resample_threshold * args.n_particles:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(args.n_particles, 1.0 / args.n_particles)

        if (k + 1) % args.print_every == 0 or (k + 1) == len(obs_sets):
            if "H" in row:
                print(
                    f"Filtered {k + 1}/{len(obs_sets)} | ESS={ess:.1f} | "
                    f"avg loglik/quote={log_pred_like_options / max(n_quotes, 1):.4f} | "
                    f"F={row['F']:.2f} A={row['A']:.6g} rho={row['rho']:.3f} "
                    f"nu={row['nu']:.3f} H={row['H']:.3f} beta={row['beta']:.3f}"
                )
            else:
                print(
                    f"Filtered {k + 1}/{len(obs_sets)} | ESS={ess:.1f} | "
                    f"avg loglik/quote={log_pred_like_options / max(n_quotes, 1):.4f} | "
                    f"F={row['F']:.2f} A={row['A']:.6g} rho={row['rho']:.3f} "
                    f"nu={row['nu']:.3f} beta={row['beta']:.3f}"
                )

    filtered_df = pd.DataFrame(filtered_rows)
    ess_df = pd.DataFrame(ess_rows)
    comparison_df = pd.concat(comparison_chunks, axis=0, ignore_index=True) if comparison_chunks else pd.DataFrame()
    loglik_df = pd.DataFrame(loglik_rows)
    if not loglik_df.empty:
        loglik_df["cumulative_log_predictive_likelihood"] = loglik_df["log_predictive_likelihood"].cumsum()
        loglik_df["cumulative_log_predictive_likelihood_with_forward_anchor"] = loglik_df["log_predictive_likelihood_with_forward_anchor"].cumsum()
        cumulative_quotes = loglik_df["n_quotes"].cumsum().clip(lower=1)
        loglik_df["cumulative_avg_log_predictive_likelihood_per_quote"] = (
            loglik_df["cumulative_log_predictive_likelihood"] / cumulative_quotes
        )
        loglik_df["cumulative_avg_log_predictive_likelihood_with_forward_anchor_per_quote"] = (
            loglik_df["cumulative_log_predictive_likelihood_with_forward_anchor"] / cumulative_quotes
        )
    rmse_df = pd.DataFrame(rmse_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    predictive_df = (
        pd.concat(predictive_chunks, axis=0, ignore_index=True)
        if predictive_chunks
        else pd.DataFrame()
    )
    return (
        filtered_df,
        ess_df,
        comparison_df,
        loglik_df,
        rmse_df,
        runtime_df,
        predictive_df,
    )



def particle_log_predictive_likelihood(weights: np.ndarray, log_likelihood: np.ndarray) -> float:
    """
    Estimate log p(y_t | y_1:t-1) from predicted particles.

    The log_likelihood values in this script intentionally omit Gaussian
    normalisation constants. This is fine for comparing models only when the
    same observations, quote weights, noise scales, and likelihood mode are used.
    """
    log_terms = np.log(np.asarray(weights, dtype=float) + 1e-300) + np.asarray(log_likelihood, dtype=float)
    m = np.nanmax(log_terms)
    if not np.isfinite(m):
        return float("-inf")
    total = np.nansum(np.exp(log_terms - m))
    if (not np.isfinite(total)) or total <= 0.0:
        return float("-inf")
    return float(m + np.log(total))


def build_model_comparison_summary(
    filtered_df: pd.DataFrame,
    loglik_df: pd.DataFrame,
    rmse_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> pd.DataFrame:
    """One-row summary for comparing runs of different models."""
    row: dict[str, float | int | str] = {}

    if not loglik_df.empty:
        n_quotes = float(loglik_df["n_quotes"].sum())
        total_ll = float(loglik_df["log_predictive_likelihood"].sum())
        total_ll_anchor = float(loglik_df["log_predictive_likelihood_with_forward_anchor"].sum())
        row.update({
            "n_timestamps": int(len(loglik_df)),
            "n_quotes_total": int(n_quotes),
            "total_log_predictive_likelihood": total_ll,
            "avg_log_predictive_likelihood_per_quote": total_ll / max(n_quotes, 1.0),
            "total_log_predictive_likelihood_with_forward_anchor": total_ll_anchor,
            "avg_log_predictive_likelihood_with_forward_anchor_per_quote": total_ll_anchor / max(n_quotes, 1.0),
            "mean_timestamp_log_predictive_likelihood": float(loglik_df["log_predictive_likelihood"].mean()),
        })

    if not filtered_df.empty:
        row.update({
            "mean_ess": float(filtered_df["ess"].mean()) if "ess" in filtered_df.columns else np.nan,
            "min_ess": float(filtered_df["ess"].min()) if "ess" in filtered_df.columns else np.nan,
        })

    if not rmse_df.empty:
        for c in ["fit_price_rmse", "fit_mean_abs_price_residual", "fit_inside_bidask_rate"]:
            if c in rmse_df.columns:
                row[f"mean_{c}"] = float(rmse_df[c].mean())

    if not runtime_df.empty:
        row.update({
            "total_runtime_seconds": float(runtime_df["timestamp_seconds"].sum()),
            "mean_timestamp_seconds": float(runtime_df["timestamp_seconds"].mean()),
            "total_predict_seconds": float(runtime_df["predict_seconds"].sum()),
            "total_likelihood_seconds": float(runtime_df["likelihood_seconds"].sum()),
            "total_update_seconds": float(runtime_df["update_seconds"].sum()),
            "total_evaluation_seconds": float(runtime_df["evaluation_seconds"].sum()),
        })

    row["loglikelihood_note"] = "Gaussian constants omitted; compare only runs using identical observations, weights, noise scales, and likelihood mode."
    return pd.DataFrame([row])

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def save_state_plot(filtered_df: pd.DataFrame, col: str, plots_dir: Path) -> None:
    if col not in filtered_df.columns:
        return
    x = filtered_df["t_index"].to_numpy(dtype=int)
    y = filtered_df[col].to_numpy(dtype=float)
    std_col = f"{col}_std"

    plt.figure(figsize=(11, 4))
    plt.plot(x, y, marker="o", markersize=3, linewidth=1.7, label=f"filtered {col}")
    if std_col in filtered_df.columns:
        s = filtered_df[std_col].to_numpy(dtype=float)
        plt.plot(x, y + s, linestyle="--", linewidth=1.1, label=f"{col} +1 std")
        plt.plot(x, y - s, linestyle="--", linewidth=1.1, label=f"{col} -1 std")
    plt.xlabel("timestamp index")
    plt.ylabel(col)
    plt.title(f"Filtered SABR state: {col}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"filtered_{safe_filename(col)}.png", dpi=160)
    plt.close()


def save_diagnostic_plots(filtered_df: pd.DataFrame, ess_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for col in ["F", "A", "atm_logvol_proxy", "rho", "nu", "beta"]:
        save_state_plot(filtered_df, col, plots_dir)

    plt.figure(figsize=(11, 4))
    x = ess_df["t_index"].to_numpy(dtype=int)
    plt.plot(x, ess_df["ess"].to_numpy(dtype=float), marker="o", markersize=3, linewidth=1.7, label="ESS")
    plt.axhline(args.resample_threshold * args.n_particles, linestyle="--", linewidth=1.1, label="resample threshold")
    plt.xlabel("timestamp index")
    plt.ylabel("ESS")
    plt.title("Effective sample size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ess.png", dpi=160)
    plt.close()

    if "fit_price_rmse" in filtered_df.columns:
        plt.figure(figsize=(11, 4))
        plt.plot(filtered_df["t_index"], filtered_df["fit_price_rmse"], marker="o", markersize=3, linewidth=1.7)
        plt.xlabel("timestamp index")
        plt.ylabel("price RMSE")
        plt.title("Filtered mean state price RMSE")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "filtered_price_rmse.png", dpi=160)
        plt.close()

        plt.figure(figsize=(11, 4))
        plt.plot(filtered_df["t_index"], filtered_df["fit_inside_bidask_rate"], marker="o", markersize=3, linewidth=1.7)
        plt.xlabel("timestamp index")
        plt.ylabel("inside bid-ask fraction")
        plt.title("Filtered mean state inside bid-ask fraction")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "filtered_inside_bidask_rate.png", dpi=160)
        plt.close()




    if "log_predictive_likelihood" in filtered_df.columns:
        plt.figure(figsize=(11, 4))
        plt.plot(filtered_df["t_index"], filtered_df["log_predictive_likelihood"], marker="o", markersize=3, linewidth=1.7)
        plt.xlabel("timestamp index")
        plt.ylabel("log predictive likelihood")
        plt.title("One-step-ahead predictive log likelihood")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "log_predictive_likelihood.png", dpi=160)
        plt.close()

        cumulative = filtered_df["log_predictive_likelihood"].cumsum()
        plt.figure(figsize=(11, 4))
        plt.plot(filtered_df["t_index"], cumulative, marker="o", markersize=3, linewidth=1.7)
        plt.xlabel("timestamp index")
        plt.ylabel("cumulative log predictive likelihood")
        plt.title("Cumulative one-step-ahead predictive log likelihood")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "cumulative_log_predictive_likelihood.png", dpi=160)
        plt.close()

        if "avg_log_predictive_likelihood_per_quote" in filtered_df.columns:
            plt.figure(figsize=(11, 4))
            plt.plot(filtered_df["t_index"], filtered_df["avg_log_predictive_likelihood_per_quote"], marker="o", markersize=3, linewidth=1.7)
            plt.xlabel("timestamp index")
            plt.ylabel("average log likelihood per quote")
            plt.title("Average predictive log likelihood per quote")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "avg_log_predictive_likelihood_per_quote.png", dpi=160)
            plt.close()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint SABR factor-parameter particle filter with nested MC likelihood.")

    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("sabr_joint_factor_parameter_pf_output"))

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-timestamps", type=int, default=100)
    parser.add_argument("--n-particles", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--print-every", type=int, default=5)

    # Raw column overrides.
    parser.add_argument("--forward-col", type=str, default="auto")
    parser.add_argument("--maturity-col", type=str, default="auto")
    parser.add_argument("--market-price-col", type=str, default="auto")

    # Option cleaning.
    parser.add_argument("--min-maturity-days", type=float, default=1.0)
    parser.add_argument("--max-maturity-days", type=float, default=120.0)
    parser.add_argument("--max-abs-log-moneyness", type=float, default=0.50)
    parser.add_argument("--max-options-per-timestamp", type=int, default=80)

    # SABR state/model.
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--estimate-beta", action="store_true")
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--rho0", type=float, default=-0.3)
    parser.add_argument("--nu0", type=float, default=1.0)
    parser.add_argument("--F-floor", type=float, default=1.0)
    parser.add_argument("--A-floor", type=float, default=1e-10)
    parser.add_argument("--nu-floor", type=float, default=1e-4)
    parser.add_argument("--nu-cap", type=float, default=10.0)

    # Initial particle spread.
    parser.add_argument("--log-F-init-sd", type=float, default=0.005)
    parser.add_argument("--log-A-init-sd", type=float, default=0.20)
    parser.add_argument("--rho-raw-init-sd", type=float, default=0.40)
    parser.add_argument("--log-nu-init-sd", type=float, default=0.50)
    parser.add_argument("--beta-raw-init-sd", type=float, default=0.25)

    # Parameter / factor market-time process noise.
    parser.add_argument("--reference-dt-years", type=float, default=5.0 / (365.0 * 24.0 * 60.0))
    parser.add_argument(
        "--log-A-process-sd",
        type=float,
        default=0.0,
        help=(
            "Additional independent random-walk standard deviation in log A "
            "per reference interval. Use 0.02 to match the rough-SABR "
            "log-U process-noise setting; use 0.0 for the original baseline."
        ),
    )
    parser.add_argument("--rho-raw-process-sd", type=float, default=0.02)
    parser.add_argument("--log-nu-process-sd", type=float, default=0.02)
    parser.add_argument("--beta-raw-process-sd", type=float, default=0.005)

    # MC pricing.
    parser.add_argument("--n-mc-paths", type=int, default=512)
    parser.add_argument("--n-eval-mc-paths", type=int, default=1024)
    parser.add_argument("--mc-days-per-step", type=float, default=2.0)
    parser.add_argument("--max-mc-steps", type=int, default=80)
    parser.add_argument("--antithetic", action="store_true", default=True)
    parser.add_argument("--no-antithetic", dest="antithetic", action="store_false")
    parser.add_argument("--payoff-chunk-size", type=int, default=32)
    parser.add_argument("--price-unit", choices=["btc", "usd"], default="btc")

    # Likelihood.
    parser.add_argument("--likelihood-components", choices=["price", "iv", "both"], default="price")
    parser.add_argument("--price-likelihood-mode", choices=["bidask-interval", "mark"], default="bidask-interval")
    parser.add_argument("--quote-weighting", choices=["equal-expiry", "equal-quote"], default="equal-expiry")
    parser.add_argument("--price-likelihood-weight", type=float, default=1.0)
    parser.add_argument("--iv-likelihood-weight", type=float, default=0.5)
    parser.add_argument("--price-abs-sd-floor", type=float, default=1e-5)
    parser.add_argument("--price-rel-sd-floor", type=float, default=0.02)
    parser.add_argument("--iv-obs-sd", type=float, default=0.01)
    parser.add_argument("--iv-inversion-iters", type=int, default=35)
    parser.add_argument("--far-log-moneyness-cutoff", type=float, default=0.35)
    parser.add_argument("--far-moneyness-multiplier", type=float, default=1.25)
    parser.add_argument("--missing-bidask-multiplier", type=float, default=1.50)
    parser.add_argument("--noise-total-max-multiplier", type=float, default=3.0)
    parser.add_argument("--forward-obs-rel-sd", type=float, default=0.002)

    # Resampling / tempering.
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--no-tempering", dest="use_tempering", action="store_false")
    parser.add_argument("--use-tempering", dest="use_tempering", action="store_true", default=True)
    parser.add_argument("--tempering-ess-fraction", type=float, default=0.70)
    parser.add_argument("--max-tempering-steps", type=int, default=40)

    # Output diagnostics.
    parser.add_argument("--evaluate-filtered-fit", action="store_true", default=True)
    parser.add_argument("--no-evaluate-filtered-fit", dest="evaluate_filtered_fit", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.n_timestamps <= 0:
        raise ValueError("--n-timestamps must be positive.")
    if args.n_particles <= 0:
        raise ValueError("--n-particles must be positive.")
    if args.n_mc_paths <= 0:
        raise ValueError("--n-mc-paths must be positive.")
    if not (args.beta_min < args.beta <= args.beta_max):
        raise ValueError("Require beta_min < beta <= beta_max.")
    if args.log_A_process_sd < 0.0:
        raise ValueError("--log-A-process-sd must be non-negative.")

    cfg = StateConfig(
        beta_fixed=args.beta,
        estimate_beta=args.estimate_beta,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        shift=args.shift,
        F_floor=args.F_floor,
        A_floor=args.A_floor,
        nu_floor=args.nu_floor,
        nu_cap=args.nu_cap,
    )

    print("Loading raw CSV...")
    raw_df = pd.read_csv(args.raw_csv)
    cols = infer_columns(raw_df, args)
    print("Detected columns:")
    print(cols)

    print("Preparing raw option observations...")
    clean_df = prepare_raw_options(raw_df, cols, args)
    times = selected_times(clean_df, args.start_index, args.n_timestamps)
    obs_sets = build_observation_sets(clean_df, times, args.start_index, args)

    print(
        f"Prepared {len(obs_sets)} timestamps starting at global index "
        f"{args.start_index}."
    )
    print(f"First timestamp options used: {len(obs_sets[0].options)}")
    print(f"Additional log-A process SD: {args.log_A_process_sd:.6g} per reference interval")
    print("Running joint SABR factor-parameter particle filter...")

    (
        filtered_df,
        ess_df,
        comparison_df,
        loglik_df,
        rmse_df,
        runtime_df,
        predictive_df,
    ) = run_filter(obs_sets, cfg, args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_dir / "filtered_state_path.csv", index=False)
    ess_df.to_csv(output_dir / "ess.csv", index=False)
    loglik_df.to_csv(output_dir / "loglikelihood.csv", index=False)
    if not rmse_df.empty:
        rmse_df.to_csv(output_dir / "surface_rmse_over_time.csv", index=False)
    runtime_df.to_csv(output_dir / "runtime_by_timestamp.csv", index=False)
    if not predictive_df.empty:
        predictive_df.to_csv(output_dir / "predictive_option_prices.csv", index=False)
    build_model_comparison_summary(filtered_df, loglik_df, rmse_df, runtime_df).to_csv(
        output_dir / "model_comparison_summary.csv", index=False
    )
    if not comparison_df.empty:
        comparison_df.to_csv(output_dir / "filtered_mean_option_fit.csv", index=False)

    # Save a copy of the selected observations for reproducibility.
    selected_obs_df = pd.concat([o.options.assign(t_index=o.t_index, selected_time=o.time_value) for o in obs_sets], ignore_index=True)
    selected_obs_df.to_csv(output_dir / "selected_observations.csv", index=False)

    save_diagnostic_plots(filtered_df, ess_df, output_dir, args)

    print("Done.")
    print(f"Filtered state path: {output_dir / 'filtered_state_path.csv'}")
    print(f"ESS diagnostics: {output_dir / 'ess.csv'}")
    print(f"Log likelihood diagnostics: {output_dir / 'loglikelihood.csv'}")
    if not rmse_df.empty:
        print(f"Surface RMSE over time: {output_dir / 'surface_rmse_over_time.csv'}")
    print(f"Runtime summary: {output_dir / 'model_comparison_summary.csv'}")
    if not predictive_df.empty:
        print(f"Predictive option prices: {output_dir / 'predictive_option_prices.csv'}")
    if not comparison_df.empty:
        print(f"Filtered option fit: {output_dir / 'filtered_mean_option_fit.csv'}")
    print(f"Plots directory: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
