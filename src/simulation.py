#!/usr/bin/env python3
"""Oracle-style simulation for knowledge-augmented penalized Cox modeling.

This script compares:
1) Baseline Cox model with uniform lasso penalty.
2) Knowledge-augmented Cox model with feature-specific lasso weights from pseudo priors.

Each model's lambda is tuned by cross-validation on the training split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class Dataset:
    x: np.ndarray
    time: np.ndarray
    event: np.ndarray
    beta_true: np.ndarray


class DualProgressBar:
    def __init__(self, total_run: int, total_cv: int, enabled: bool = True) -> None:
        self.total_run = max(1, int(total_run))
        self.total_cv = max(1, int(total_cv))
        self.done_run = 0
        self.done_cv = 0
        self.enabled = enabled
        self._last_render = 0.0
        if self.enabled:
            self._render(force=True)

    @staticmethod
    def _bar(done: int, total: int, width: int = 16) -> str:
        frac = min(1.0, max(0.0, done / total))
        filled = int(width * frac)
        return "[" + ("#" * filled) + ("." * (width - filled)) + "]"

    def _render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_render) < 0.05:
            return
        self._last_render = now
        msg = (
            f"\rRun {self._bar(self.done_run, self.total_run)} {self.done_run}/{self.total_run}  "
            f"CV {self._bar(self.done_cv, self.total_cv)} {self.done_cv}/{self.total_cv}"
        )
        print(msg, end="", file=sys.stdout, flush=True)

    def update(self, run: int = 0, cv: int = 0) -> None:
        if not self.enabled:
            return
        self.done_run = min(self.total_run, self.done_run + int(run))
        self.done_cv = min(self.total_cv, self.done_cv + int(cv))
        self._render(force=False)

    def close(self) -> None:
        if not self.enabled:
            return
        self._render(force=True)
        print("", file=sys.stdout, flush=True)


def make_block_covariance(p: int, group_size: int, within_corr: float) -> np.ndarray:
    cov = np.eye(p)
    if within_corr <= 0:
        return cov
    n_groups = int(np.ceil(p / group_size))
    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, p)
        block_len = end - start
        if block_len <= 1:
            continue
        block = np.full((block_len, block_len), within_corr)
        np.fill_diagonal(block, 1.0)
        cov[start:end, start:end] = block
    return cov


def generate_synthetic_cox_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    n_active: int,
    group_size: int,
    within_corr: float,
    baseline_hazard: float,
    censoring_hazard: float,
    effect_low: float,
    effect_high: float,
) -> Dataset:
    cov = make_block_covariance(n_features, group_size, within_corr)
    x = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    beta_true = np.zeros(n_features)
    active_idx = rng.choice(n_features, size=n_active, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_active)
    magnitudes = rng.uniform(effect_low, effect_high, size=n_active)
    beta_true[active_idx] = signs * magnitudes

    linpred = x @ beta_true
    u = rng.uniform(size=n_samples)
    event_time = -np.log(u) / (baseline_hazard * np.exp(np.clip(linpred, -20, 20)))

    censor_time = rng.exponential(scale=1.0 / censoring_hazard, size=n_samples)
    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.int64)

    return Dataset(x=x, time=observed_time, event=event, beta_true=beta_true)


def generate_knowledge_scores(
    beta_true: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    noise_kind: str = "uniform",
) -> np.ndarray:
    active = (np.abs(beta_true) > 0).astype(np.float64)
    if noise_kind == "uniform":
        noise = rng.uniform(0.0, 1.0, size=beta_true.shape[0])
    elif noise_kind == "normal":
        noise = rng.normal(loc=0.5, scale=0.2, size=beta_true.shape[0])
        noise = np.clip(noise, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported noise_kind: {noise_kind}")

    scores = rho * active + (1.0 - rho) * noise
    return np.clip(scores, 0.0, 1.0)


def weights_from_scores(scores: np.ndarray, alpha: float, eps: float = 1e-6) -> np.ndarray:
    weights = np.exp(-alpha * scores)
    return np.maximum(weights, eps)


def _cox_sort(time: np.ndarray, event: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(time)[::-1]
    return time[order], event[order], x[order]


def cox_negloglik_and_grad(
    beta: np.ndarray,
    time_sorted: np.ndarray,
    event_sorted: np.ndarray,
    x_sorted: np.ndarray,
) -> Tuple[float, np.ndarray]:
    del time_sorted  # sorted order already encodes risk sets

    event_idx = np.where(event_sorted == 1)[0]
    n_events = event_idx.shape[0]
    if n_events == 0:
        raise ValueError("No observed events in training data; cannot fit Cox model.")

    eta = x_sorted @ beta
    exp_eta = np.exp(np.clip(eta, -50.0, 50.0))

    s0 = np.cumsum(exp_eta)
    s1 = np.cumsum(exp_eta[:, None] * x_sorted, axis=0)

    s0_event = s0[event_idx]
    s1_event = s1[event_idx]
    x_event = x_sorted[event_idx]

    loss = -np.sum(eta[event_idx] - np.log(s0_event)) / n_events
    grad = -np.sum(x_event - (s1_event / s0_event[:, None]), axis=0) / n_events
    return float(loss), grad


def cox_negloglik(
    beta: np.ndarray,
    time_sorted: np.ndarray,
    event_sorted: np.ndarray,
    x_sorted: np.ndarray,
) -> float:
    loss, _ = cox_negloglik_and_grad(beta, time_sorted, event_sorted, x_sorted)
    return loss


def soft_threshold(z: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def fit_weighted_lasso_cox(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    lam: float,
    penalty_weights: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    step_init: float = 1.0,
) -> np.ndarray:
    time_sorted, event_sorted, x_sorted = _cox_sort(time, event, x)
    p = x.shape[1]
    beta = np.zeros(p)
    step = step_init

    for _ in range(max_iter):
        smooth_loss, grad = cox_negloglik_and_grad(beta, time_sorted, event_sorted, x_sorted)
        obj = smooth_loss + lam * np.sum(penalty_weights * np.abs(beta))

        accepted = False
        for _ in range(30):
            z = beta - step * grad
            beta_new = soft_threshold(z, step * lam * penalty_weights)

            smooth_new = cox_negloglik(beta_new, time_sorted, event_sorted, x_sorted)
            obj_new = smooth_new + lam * np.sum(penalty_weights * np.abs(beta_new))

            delta = beta_new - beta
            rhs = smooth_loss + grad @ delta + (np.linalg.norm(delta) ** 2) / (2.0 * step)
            if smooth_new <= rhs + 1e-12:
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

        if np.linalg.norm(beta_new - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break

        beta = beta_new
        step = min(step * 1.05, step_init)

        # Optional monotonic safeguard for numerical issues.
        if obj_new > obj + 1e-8:
            step *= 0.5

    return beta


def concordance_index_harrell(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    n = len(time)
    concordant = 0.0
    comparable = 0.0

    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[i] >= time[j]:
                continue
            comparable += 1.0
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5

    if comparable == 0:
        return float("nan")
    return concordant / comparable


def breslow_baseline_cumhaz(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    event_times = np.unique(time[event == 1])
    if event_times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    increments = np.zeros_like(event_times, dtype=float)
    for i, t in enumerate(event_times):
        d_t = np.sum((time == t) & (event == 1))
        denom = np.sum(risk[time >= t])
        increments[i] = 0.0 if denom <= 0 else d_t / denom
    return event_times, np.cumsum(increments)


def step_function_eval(x: np.ndarray, y: np.ndarray, q: float, side: str = "right", default: float = 1.0) -> float:
    if x.size == 0:
        return default
    idx = np.searchsorted(x, q, side=side) - 1
    if idx < 0:
        return default
    return float(y[idx])


def predict_survival_at_time(
    x: np.ndarray, beta: np.ndarray, event_times: np.ndarray, cumhaz: np.ndarray, t_eval: float
) -> np.ndarray:
    h0_t = step_function_eval(event_times, cumhaz, t_eval, side="right", default=0.0)
    lp = x @ beta
    hr = np.exp(np.clip(lp, -50.0, 50.0))
    return np.exp(-h0_t * hr)


def kaplan_meier_survival(time: np.ndarray, event_indicator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_times = np.unique(time)
    surv = np.ones(unique_times.shape[0], dtype=float)
    s_t = 1.0
    for i, t in enumerate(unique_times):
        n_risk = np.sum(time >= t)
        d_t = np.sum((time == t) & (event_indicator == 1))
        if n_risk > 0:
            s_t *= 1.0 - (d_t / n_risk)
        surv[i] = s_t
    return unique_times, surv


def ipcw_brier_score(
    time: np.ndarray,
    event: np.ndarray,
    surv_prob_at_t: np.ndarray,
    t_eval: float,
    censor_km_time: np.ndarray,
    censor_km_surv: np.ndarray,
    eps: float = 1e-6,
) -> float:
    y = (time > t_eval).astype(float)
    weights = np.zeros_like(time, dtype=float)

    for i in range(time.shape[0]):
        if time[i] <= t_eval and event[i] == 1:
            g = step_function_eval(censor_km_time, censor_km_surv, float(time[i]), side="left", default=1.0)
            weights[i] = 1.0 / max(g, eps)
        elif time[i] > t_eval:
            g = step_function_eval(censor_km_time, censor_km_surv, float(t_eval), side="right", default=1.0)
            weights[i] = 1.0 / max(g, eps)
        else:
            weights[i] = 0.0

    return float(np.mean(weights * (y - surv_prob_at_t) ** 2))


def integrated_brier_score(
    time: np.ndarray,
    event: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    baseline_event_times: np.ndarray,
    baseline_cumhaz: np.ndarray,
    time_grid: np.ndarray,
    censor_km_time: np.ndarray,
    censor_km_surv: np.ndarray,
) -> float:
    if time_grid.size == 0:
        return float("nan")

    bs_values = []
    for t_eval in time_grid:
        surv_prob = predict_survival_at_time(x, beta, baseline_event_times, baseline_cumhaz, float(t_eval))
        bs = ipcw_brier_score(
            time=time,
            event=event,
            surv_prob_at_t=surv_prob,
            t_eval=float(t_eval),
            censor_km_time=censor_km_time,
            censor_km_surv=censor_km_surv,
        )
        bs_values.append(bs)

    bs_arr = np.asarray(bs_values, dtype=float)
    if time_grid.size == 1 or float(time_grid[-1] - time_grid[0]) <= 0.0:
        return float(np.mean(bs_arr))
    return float(np.trapezoid(bs_arr, time_grid) / (time_grid[-1] - time_grid[0]))


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def parse_float_list(values: str) -> List[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


def build_cv_folds(n_samples: int, n_folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    n_folds = max(2, min(n_folds, n_samples))
    perm = rng.permutation(n_samples)
    folds = np.array_split(perm, n_folds)
    return [fold.astype(int) for fold in folds if fold.size > 0]


def tune_lambda_via_cv(
    x_raw: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    penalty_weights: np.ndarray,
    lam_grid: Sequence[float],
    cv_folds: Sequence[np.ndarray],
    max_iter: int,
    tol: float,
    fallback_lam: float,
    progress: Optional[DualProgressBar] = None,
) -> float:
    if len(cv_folds) < 2:
        return fallback_lam

    n = x_raw.shape[0]
    best_lam = fallback_lam
    best_score = -np.inf

    for lam in lam_grid:
        fold_scores: List[float] = []
        for val_idx in cv_folds:
            if progress is not None:
                progress.update(cv=1)
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            tr_idx = np.where(train_mask)[0]
            if tr_idx.size == 0:
                continue

            x_tr_raw = x_raw[tr_idx]
            t_tr = time[tr_idx]
            e_tr = event[tr_idx]
            x_va_raw = x_raw[val_idx]
            t_va = time[val_idx]
            e_va = event[val_idx]

            if np.sum(e_tr) == 0 or np.sum(e_va) == 0:
                continue

            x_tr, x_va = standardize_train_test(x_tr_raw, x_va_raw)
            try:
                beta = fit_weighted_lasso_cox(
                    x=x_tr,
                    time=t_tr,
                    event=e_tr,
                    lam=float(lam),
                    penalty_weights=penalty_weights,
                    max_iter=max_iter,
                    tol=tol,
                )
            except ValueError:
                continue

            score = concordance_index_harrell(t_va, e_va, x_va @ beta)
            if not np.isnan(score):
                fold_scores.append(float(score))

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_score > best_score + 1e-12:
            best_score = mean_score
            best_lam = float(lam)

    return best_lam


def run_experiment(args: argparse.Namespace) -> Tuple[Dict[Tuple[float, int], Dict[str, float]], List[int]]:
    rng_master = np.random.default_rng(args.seed)
    train_sizes = [int(v) for v in args.train_sizes.split(",")]
    rhos = parse_float_list(args.rho_list)
    lam_grid = sorted(set(parse_float_list(args.lam_grid)))
    if not lam_grid:
        lam_grid = [float(args.lam)]
    repeat_seeds = [int(s) for s in rng_master.integers(0, 2**32 - 1, size=args.n_repeats)]
    max_train = max(train_sizes)

    if max_train >= args.n_samples:
        raise ValueError("max(train_sizes) must be smaller than n_samples to keep a test split.")

    results: Dict[Tuple[float, int], Dict[str, List[float]]] = {}
    for rho in rhos:
        for n_train in train_sizes:
            results[(rho, n_train)] = {
                "baseline_cindex": [],
                "knowledge_cindex": [],
                "baseline_brier": [],
                "knowledge_brier": [],
                "baseline_ibs": [],
                "knowledge_ibs": [],
                "baseline_lambda": [],
                "knowledge_lambda": [],
            }

    cv_steps_per_split = {
        n_train: 2 * len(lam_grid) * max(2, min(args.cv_folds, n_train)) for n_train in train_sizes
    }
    total_run_steps = args.n_repeats * len(rhos) * len(train_sizes)
    total_cv_steps = args.n_repeats * len(rhos) * sum(cv_steps_per_split[n] for n in train_sizes)
    progress = DualProgressBar(total_run_steps, total_cv_steps, enabled=(not args.no_progress))

    try:
        for rep_seed in repeat_seeds:
            rng = np.random.default_rng(rep_seed)
            data = generate_synthetic_cox_data(
                rng=rng,
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_active=args.n_active,
                group_size=args.group_size,
                within_corr=args.within_corr,
                baseline_hazard=args.baseline_hazard,
                censoring_hazard=args.censoring_hazard,
                effect_low=args.effect_low,
                effect_high=args.effect_high,
            )

            perm = rng.permutation(args.n_samples)
            train_pool_idx = perm[:max_train]
            test_idx = perm[max_train:]

            x_pool = data.x[train_pool_idx]
            t_pool = data.time[train_pool_idx]
            e_pool = data.event[train_pool_idx]

            x_test_raw = data.x[test_idx]
            t_test = data.time[test_idx]
            e_test = data.event[test_idx]

            for rho in rhos:
                scores = generate_knowledge_scores(data.beta_true, rho=rho, rng=rng, noise_kind=args.noise_kind)
                knowledge_weights = weights_from_scores(scores, alpha=args.alpha)
                baseline_weights = np.ones(args.n_features, dtype=float)

                order = rng.permutation(max_train)
                for n_train in train_sizes:
                    idx = order[:n_train]
                    x_train_raw = x_pool[idx]
                    t_train = t_pool[idx]
                    e_train = e_pool[idx]

                    cv_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                    cv_folds = build_cv_folds(n_train, args.cv_folds, cv_rng)
                    expected_cv_steps_split = 2 * len(lam_grid) * len(cv_folds)

                    # Skip rare degenerate split with no events.
                    if np.sum(e_train) == 0:
                        progress.update(cv=expected_cv_steps_split, run=1)
                        continue

                    best_lam_base = tune_lambda_via_cv(
                        x_raw=x_train_raw,
                        time=t_train,
                        event=e_train,
                        penalty_weights=baseline_weights,
                        lam_grid=lam_grid,
                        cv_folds=cv_folds,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        fallback_lam=args.lam,
                        progress=progress,
                    )
                    best_lam_know = tune_lambda_via_cv(
                        x_raw=x_train_raw,
                        time=t_train,
                        event=e_train,
                        penalty_weights=knowledge_weights,
                        lam_grid=lam_grid,
                        cv_folds=cv_folds,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        fallback_lam=args.lam,
                        progress=progress,
                    )

                    x_train, x_test = standardize_train_test(x_train_raw, x_test_raw)

                    beta_base = fit_weighted_lasso_cox(
                        x=x_train,
                        time=t_train,
                        event=e_train,
                        lam=best_lam_base,
                        penalty_weights=baseline_weights,
                        max_iter=args.max_iter,
                        tol=args.tol,
                    )

                    beta_know = fit_weighted_lasso_cox(
                        x=x_train,
                        time=t_train,
                        event=e_train,
                        lam=best_lam_know,
                        penalty_weights=knowledge_weights,
                        max_iter=args.max_iter,
                        tol=args.tol,
                    )

                    risk_base = x_test @ beta_base
                    risk_know = x_test @ beta_know

                    c_base = concordance_index_harrell(t_test, e_test, risk_base)
                    c_know = concordance_index_harrell(t_test, e_test, risk_know)

                    risk_train_base = np.exp(np.clip(x_train @ beta_base, -50.0, 50.0))
                    risk_train_know = np.exp(np.clip(x_train @ beta_know, -50.0, 50.0))
                    base_evt_t, base_cumhaz = breslow_baseline_cumhaz(t_train, e_train, risk_train_base)
                    know_evt_t, know_cumhaz = breslow_baseline_cumhaz(t_train, e_train, risk_train_know)

                    c_km_t, c_km_s = kaplan_meier_survival(t_test, 1 - e_test)

                    train_event_times = t_train[e_train == 1]
                    if train_event_times.size > 0:
                        t_eval = float(np.quantile(train_event_times, args.brier_time_quantile))
                        time_grid = np.quantile(
                            train_event_times,
                            np.linspace(args.ibs_min_quantile, args.ibs_max_quantile, args.ibs_num_times),
                        )
                        time_grid = np.unique(time_grid)
                    else:
                        t_eval = float(np.quantile(t_train, args.brier_time_quantile))
                        time_grid = np.array([t_eval], dtype=float)

                    s_base_t = predict_survival_at_time(x_test, beta_base, base_evt_t, base_cumhaz, t_eval)
                    s_know_t = predict_survival_at_time(x_test, beta_know, know_evt_t, know_cumhaz, t_eval)
                    brier_base = ipcw_brier_score(
                        time=t_test,
                        event=e_test,
                        surv_prob_at_t=s_base_t,
                        t_eval=t_eval,
                        censor_km_time=c_km_t,
                        censor_km_surv=c_km_s,
                    )
                    brier_know = ipcw_brier_score(
                        time=t_test,
                        event=e_test,
                        surv_prob_at_t=s_know_t,
                        t_eval=t_eval,
                        censor_km_time=c_km_t,
                        censor_km_surv=c_km_s,
                    )

                    ibs_base = integrated_brier_score(
                        time=t_test,
                        event=e_test,
                        x=x_test,
                        beta=beta_base,
                        baseline_event_times=base_evt_t,
                        baseline_cumhaz=base_cumhaz,
                        time_grid=time_grid,
                        censor_km_time=c_km_t,
                        censor_km_surv=c_km_s,
                    )
                    ibs_know = integrated_brier_score(
                        time=t_test,
                        event=e_test,
                        x=x_test,
                        beta=beta_know,
                        baseline_event_times=know_evt_t,
                        baseline_cumhaz=know_cumhaz,
                        time_grid=time_grid,
                        censor_km_time=c_km_t,
                        censor_km_surv=c_km_s,
                    )

                    bucket = results[(rho, n_train)]
                    bucket["baseline_cindex"].append(c_base)
                    bucket["knowledge_cindex"].append(c_know)
                    bucket["baseline_brier"].append(brier_base)
                    bucket["knowledge_brier"].append(brier_know)
                    bucket["baseline_ibs"].append(ibs_base)
                    bucket["knowledge_ibs"].append(ibs_know)
                    bucket["baseline_lambda"].append(best_lam_base)
                    bucket["knowledge_lambda"].append(best_lam_know)
                    progress.update(run=1)
    finally:
        progress.close()

    summary: Dict[Tuple[float, int], Dict[str, float]] = {}
    for key, vals in results.items():
        c_base_mean, c_base_std = summarize(vals["baseline_cindex"])
        c_know_mean, c_know_std = summarize(vals["knowledge_cindex"])
        b_base_mean, b_base_std = summarize(vals["baseline_brier"])
        b_know_mean, b_know_std = summarize(vals["knowledge_brier"])
        ibs_base_mean, ibs_base_std = summarize(vals["baseline_ibs"])
        ibs_know_mean, ibs_know_std = summarize(vals["knowledge_ibs"])
        lam_base_mean, lam_base_std = summarize(vals["baseline_lambda"])
        lam_know_mean, lam_know_std = summarize(vals["knowledge_lambda"])
        summary[key] = {
            "baseline_cindex_mean": c_base_mean,
            "baseline_cindex_std": c_base_std,
            "knowledge_cindex_mean": c_know_mean,
            "knowledge_cindex_std": c_know_std,
            "baseline_brier_mean": b_base_mean,
            "baseline_brier_std": b_base_std,
            "knowledge_brier_mean": b_know_mean,
            "knowledge_brier_std": b_know_std,
            "baseline_ibs_mean": ibs_base_mean,
            "baseline_ibs_std": ibs_base_std,
            "knowledge_ibs_mean": ibs_know_mean,
            "knowledge_ibs_std": ibs_know_std,
            "baseline_lambda_mean": lam_base_mean,
            "baseline_lambda_std": lam_base_std,
            "knowledge_lambda_mean": lam_know_mean,
            "knowledge_lambda_std": lam_know_std,
            "n_runs": len(vals["baseline_cindex"]),
        }
    return summary, repeat_seeds


def print_summary(summary: Dict[Tuple[float, int], Dict[str, float]], train_sizes: Iterable[int], rhos: Iterable[float]) -> None:
    header = (
        "rho   n_train  baseline_cidx(mean±sd)  knowledge_cidx(mean±sd)  "
        "baseline_brier(mean±sd)  knowledge_brier(mean±sd)  baseline_ibs(mean±sd)  knowledge_ibs(mean±sd)  "
        "baseline_lam(mean±sd)  knowledge_lam(mean±sd)"
    )
    print(header)
    print("-" * len(header))

    for rho in rhos:
        for n_train in train_sizes:
            s = summary[(rho, n_train)]
            print(
                f"{rho:0.2f}  {n_train:7d}  "
                f"{s['baseline_cindex_mean']:.4f}±{s['baseline_cindex_std']:.4f}      "
                f"{s['knowledge_cindex_mean']:.4f}±{s['knowledge_cindex_std']:.4f}      "
                f"{s['baseline_brier_mean']:.4f}±{s['baseline_brier_std']:.4f}      "
                f"{s['knowledge_brier_mean']:.4f}±{s['knowledge_brier_std']:.4f}      "
                f"{s['baseline_ibs_mean']:.4f}±{s['baseline_ibs_std']:.4f}      "
                f"{s['knowledge_ibs_mean']:.4f}±{s['knowledge_ibs_std']:.4f}      "
                f"{s['baseline_lambda_mean']:.4f}±{s['baseline_lambda_std']:.4f}      "
                f"{s['knowledge_lambda_mean']:.4f}±{s['knowledge_lambda_std']:.4f}"
            )


def save_summary_csv(
    summary: Dict[Tuple[float, int], Dict[str, float]],
    output_path: Path,
    train_sizes: Iterable[int],
    rhos: Iterable[float],
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "rho",
                "n_train",
                "baseline_cindex_mean",
                "baseline_cindex_std",
                "knowledge_cindex_mean",
                "knowledge_cindex_std",
                "baseline_brier_mean",
                "baseline_brier_std",
                "knowledge_brier_mean",
                "knowledge_brier_std",
                "baseline_ibs_mean",
                "baseline_ibs_std",
                "knowledge_ibs_mean",
                "knowledge_ibs_std",
                "baseline_lambda_mean",
                "baseline_lambda_std",
                "knowledge_lambda_mean",
                "knowledge_lambda_std",
                "n_runs",
            ]
        )
        for rho in rhos:
            for n_train in train_sizes:
                s = summary[(rho, n_train)]
                writer.writerow(
                    [
                        seed,
                        f"{rho:.4f}",
                        n_train,
                        f"{s['baseline_cindex_mean']:.4f}",
                        f"{s['baseline_cindex_std']:.4f}",
                        f"{s['knowledge_cindex_mean']:.4f}",
                        f"{s['knowledge_cindex_std']:.4f}",
                        f"{s['baseline_brier_mean']:.4f}",
                        f"{s['baseline_brier_std']:.4f}",
                        f"{s['knowledge_brier_mean']:.4f}",
                        f"{s['knowledge_brier_std']:.4f}",
                        f"{s['baseline_ibs_mean']:.4f}",
                        f"{s['baseline_ibs_std']:.4f}",
                        f"{s['knowledge_ibs_mean']:.4f}",
                        f"{s['knowledge_ibs_std']:.4f}",
                        f"{s['baseline_lambda_mean']:.4f}",
                        f"{s['baseline_lambda_std']:.4f}",
                        f"{s['knowledge_lambda_mean']:.4f}",
                        f"{s['knowledge_lambda_std']:.4f}",
                        s["n_runs"],
                    ]
                )


def save_summary_plot(
    summary: Dict[Tuple[float, int], Dict[str, float]],
    output_path: Path,
    train_sizes: Sequence[int],
    rhos: Sequence[float],
    show_ci: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(train_sizes, dtype=float)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(rhos), 3)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    panels = [
        ("C-index", "baseline_cindex_mean", "baseline_cindex_std", "knowledge_cindex_mean", "knowledge_cindex_std"),
        ("Brier score", "baseline_brier_mean", "baseline_brier_std", "knowledge_brier_mean", "knowledge_brier_std"),
        ("IBS", "baseline_ibs_mean", "baseline_ibs_std", "knowledge_ibs_mean", "knowledge_ibs_std"),
        (
            "Tuned lambda",
            "baseline_lambda_mean",
            "baseline_lambda_std",
            "knowledge_lambda_mean",
            "knowledge_lambda_std",
        ),
    ]

    for ax, (title, base_m, base_s, know_m, know_s) in zip(axes.flatten(), panels):
        for i, rho in enumerate(rhos):
            color = colors[i]
            y_base = np.array([summary[(rho, n)][base_m] for n in train_sizes], dtype=float)
            sd_base = np.array([summary[(rho, n)][base_s] for n in train_sizes], dtype=float)
            y_know = np.array([summary[(rho, n)][know_m] for n in train_sizes], dtype=float)
            sd_know = np.array([summary[(rho, n)][know_s] for n in train_sizes], dtype=float)
            n_runs = np.array([summary[(rho, n)]["n_runs"] for n in train_sizes], dtype=float)

            # 95% CI half-width using normal approximation: 1.96 * SD / sqrt(n).
            e_base = np.where(
                (n_runs > 1) & np.isfinite(sd_base),
                1.96 * sd_base / np.sqrt(n_runs),
                0.0,
            )
            e_know = np.where(
                (n_runs > 1) & np.isfinite(sd_know),
                1.96 * sd_know / np.sqrt(n_runs),
                0.0,
            )

            ax.plot(x, y_base, "--o", color=color, linewidth=2.0, markersize=4)
            if show_ci:
                ax.fill_between(x, y_base - e_base, y_base + e_base, color=color, alpha=0.10)
            ax.plot(x, y_know, "-s", color=color, linewidth=2.0, markersize=4)
            if show_ci:
                ax.fill_between(x, y_know - e_know, y_know + e_know, color=color, alpha=0.18)

        ax.set_title(title)
        ax.set_xlabel("Training sample size")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.set_xticks(x)

    rho_handles = [Line2D([0], [0], color=colors[i], lw=2.4) for i in range(len(rhos))]
    rho_labels = [f"rho={rho:.2f}" for rho in rhos]
    style_handles = [
        Line2D([0], [0], color="black", lw=2.6, linestyle="--", marker="o", markersize=5),
        Line2D([0], [0], color="black", lw=2.6, linestyle="-", marker="s", markersize=5),
    ]
    style_labels = ["Baseline (dashed)", "Prior-guided (solid)"]

    fig.legend(
        style_handles,
        style_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.legend(
        rho_handles,
        rho_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=max(1, min(5, len(rhos))),
        frameon=False,
        fontsize=9,
    )
    title = "Knowledge-guided vs Baseline Cox (CV-tuned lambda)"
    if show_ci:
        title += ", shaded=95% CI"
    fig.suptitle(title, y=0.98, fontsize=13)
    fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_repro_metadata(
    args: argparse.Namespace,
    output_path: Path,
    repeat_seeds: Sequence[int],
    train_sizes: Sequence[int],
    rhos: Sequence[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "master_seed": int(args.seed),
        "repeat_seeds": [int(s) for s in repeat_seeds],
        "n_repeats": int(args.n_repeats),
        "train_sizes": [int(x) for x in train_sizes],
        "rhos": [float(r) for r in rhos],
        "lam_grid": parse_float_list(args.lam_grid) if args.lam_grid else [float(args.lam)],
        "cv_folds": int(args.cv_folds),
        "n_samples": int(args.n_samples),
        "n_features": int(args.n_features),
        "n_active": int(args.n_active),
        "alpha": float(args.alpha),
        "noise_kind": str(args.noise_kind),
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulation for knowledge-augmented penalized Cox models")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-features", type=int, default=100)
    p.add_argument("--n-active", type=int, default=10)
    p.add_argument("--train-sizes", type=str, default="200,400,600,800")
    p.add_argument("--rho-list", type=str, default="1.0,0.95,0.9,0.8,0.7")
    p.add_argument(
        "--lam",
        type=float,
        default=0.03,
        help="Fallback lambda if CV fails on a split",
    )
    p.add_argument(
        "--lam-grid",
        type=str,
        default="0.003,0.01,0.03,0.1,0.3",
        help="Candidate lambdas for CV tuning (comma-separated)",
    )
    p.add_argument("--cv-folds", type=int, default=3, help="Number of folds for lambda tuning")
    p.add_argument("--alpha", type=float, default=2.0, help="Knowledge weight strength: w_j=exp(-alpha*s_j)")
    p.add_argument("--n-repeats", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--group-size", type=int, default=10)
    p.add_argument("--within-corr", type=float, default=0.25)
    p.add_argument("--baseline-hazard", type=float, default=0.02)
    p.add_argument("--censoring-hazard", type=float, default=0.01)
    p.add_argument("--effect-low", type=float, default=0.5)
    p.add_argument("--effect-high", type=float, default=1.0)
    p.add_argument("--brier-time-quantile", type=float, default=0.5)
    p.add_argument("--ibs-min-quantile", type=float, default=0.1)
    p.add_argument("--ibs-max-quantile", type=float, default=0.9)
    p.add_argument("--ibs-num-times", type=int, default=15)
    p.add_argument("--noise-kind", type=str, default="uniform", choices=["uniform", "normal"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    p.add_argument("--no-plot-ci", action="store_true", help="Disable CI shading in output plot")
    p.add_argument("--output-csv", type=str, default="results/simulation_summary.csv")
    p.add_argument("--output-png", type=str, default="results/simulation_summary.png")
    p.add_argument("--output-metadata", type=str, default="results/simulation_metadata.json")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train_sizes = [int(v) for v in args.train_sizes.split(",")]
    rhos = parse_float_list(args.rho_list)

    summary, repeat_seeds = run_experiment(args)
    print_summary(summary, train_sizes=train_sizes, rhos=rhos)

    out = Path(args.output_csv)
    save_summary_csv(summary, out, train_sizes=train_sizes, rhos=rhos, seed=args.seed)
    out_png = Path(args.output_png)
    save_summary_plot(summary, out_png, train_sizes=train_sizes, rhos=rhos, show_ci=(not args.no_plot_ci))
    out_meta = Path(args.output_metadata)
    save_repro_metadata(args, out_meta, repeat_seeds, train_sizes=train_sizes, rhos=rhos)
    print(f"\nSaved summary to: {out}")
    print(f"Saved figure to: {out_png}")
    print(f"Saved metadata to: {out_meta}")


if __name__ == "__main__":
    main()
