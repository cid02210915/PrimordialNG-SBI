"""SBI diagnostics moved from run_pipeline(3).ipynb.

Plotting functions read previously generated results. The two explicitly named
compute_* functions perform the notebook's SBC and Knox numerical tests once.
No function in this module trains a posterior.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PARAMS = [r"$f_{\rm NL}$", r"$b_v$"]

def _root(pipe):
    return Path(pipe.settings.data_root)

def _predictions(root):
    samples = np.load(root / "posterior_samples.npz")["samples"]
    true = np.load(root / "heldout_data.npz")["theta_test"]
    return samples, true

def _finish(fig, path, save):
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


# 01 / Posterior precision, and 06 / Fisher benchmark
# The source notebook drew both in the same 3x3 contour figure.
def plot_posterior(pipe, n_show=9, indices=None, fisher=False, save=False):
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde, chi2

    cfg = pipe.settings
    samples_all, theta_all = _predictions(_root(pipe))
    ids = list(range(min(n_show, len(theta_all)))) if indices is None else list(indices)
    cols = int(np.ceil(np.sqrt(len(ids))))
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows), squeeze=False)

    if fisher:
        from .theory import load_theory, make_theory_cls
        cls = load_theory(cfg)

        def cov(theta):
            gg, vg, vv = make_theory_cls(*theta, cls, cap=cfg.cap)
            top = np.concatenate([gg, vg.transpose(1, 0, 2)], axis=1)
            bottom = np.concatenate([vg, vv], axis=1)
            return np.concatenate([top, bottom], axis=0)

        def fisher_cov(theta):
            C = cov(theta)
            dC = [
                (cov(theta + h * np.eye(2)[i]) - cov(theta - h * np.eye(2)[i])) / (2 * h)
                for i, h in enumerate([1.0, 0.001])
            ]
            F = np.zeros((2, 2))
            for ell in range(max(2, cfg.ell_min), cfg.lmax + 1):
                A = [np.linalg.solve(C[:, :, ell], d[:, :, ell]) for d in dC]
                F += (2 * ell + 1) / 2 * np.array([
                    [np.trace(a @ b) for b in A] for a in A
                ])
            return np.linalg.inv(F)

    for j, ax in zip(ids, axes.flat):
        samples = samples_all[j]
        true = theta_all[j]
        median = np.median(samples, axis=0)

        if fisher:
            Sigma = fisher_cov(true)
            std = np.sqrt(np.diag(Sigma))
            lo = np.minimum(np.percentile(samples, 1, axis=0), true - 3.5 * std)
            hi = np.maximum(np.percentile(samples, 99, axis=0), true + 3.5 * std)
        else:
            lo = np.minimum(np.percentile(samples, 1, axis=0), true)
            hi = np.maximum(np.percentile(samples, 99, axis=0), true)
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 150),
                           np.linspace(lo[1], hi[1], 150))
        Z = gaussian_kde(samples.T)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        z = np.sort(Z.ravel())[::-1]
        cum = np.cumsum(z) / z.sum()
        l95, l68 = [z[np.searchsorted(cum, p)] for p in (0.95, 0.68)]
        ax.contourf(X, Y, Z, levels=[l68, Z.max()], colors=["#dbeaf4"], alpha=0.8)
        ax.contour(X, Y, Z, levels=[l95, l68],
                   colors=["tab:blue", "tab:blue"], linewidths=[1.5, 2.0])

        if fisher:
            vals, vecs = np.linalg.eigh(Sigma)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            for p in (0.68, 0.95):
                r = np.sqrt(chi2.ppf(p, df=2))
                ax.add_patch(Ellipse(
                    xy=true, width=2 * r * np.sqrt(vals[0]),
                    height=2 * r * np.sqrt(vals[1]), angle=angle,
                    fill=False, edgecolor="tab:green", linestyle="--",
                    linewidth=1.8, zorder=4,
                ))

        ax.scatter(*true, color="tab:red", marker="*", s=110,
                   edgecolors="white", linewidths=0.7, zorder=5)
        ax.scatter(*median, color="tab:orange", marker="o", s=55,
                   edgecolors="white", linewidths=0.8, zorder=6)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
               xlabel=PARAMS[0], ylabel=PARAMS[1],
               title=f"Held-out realisation {j}")
        ax.grid(alpha=0.4)

    for ax in axes.flat[len(ids):]:
        ax.set_visible(False)

    legend = [
        Line2D([0], [0], color="tab:blue", lw=2, label="SBI 68% / 95%"),
        Line2D([0], [0], marker="*", linestyle="None", color="tab:red",
               markersize=10, label="True parameters"),
        Line2D([0], [0], marker="o", linestyle="None", color="tab:orange",
               markersize=7, label="SBI median"),
    ]
    if fisher:
        legend.insert(1, Line2D([0], [0], color="tab:green", lw=2,
                                ls="--", label="Fisher 68% / 95%"))
    fig.legend(handles=legend, loc="upper center", ncol=len(legend),
               frameon=False, bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _finish(fig, _root(pipe) / "diagnostics" / ("fisher.png" if fisher else "posterior.png"), save)

def plot_fisher(pipe, n_show=9, indices=None, save=False):
    return plot_posterior(pipe, n_show=n_show, indices=indices, fisher=True, save=save)


# 02 / Recovery: true versus median and residuals, with binned bootstrap CI.
def plot_recovery(pipe, parameter=None, nboot=1000, save=False):
    samples, true = _predictions(_root(pipe))
    inferred = np.median(samples, axis=1)
    cfg = pipe.settings
    rng = np.random.default_rng(12345)
    names = PARAMS
    res_names = [r"$\hat f_{\rm NL}-f_{\rm NL}^{\rm true}$",
                 r"$\hat b_v-b_v^{\rm true}$"]
    bounds = [cfg.fNL_prior, cfg.bv_prior]
    selected = [0, 1] if parameter is None else [dict(fNL=0, bv=1)[parameter]]
    fig, axes = plt.subplots(2, len(selected), figsize=(6 * len(selected), 9),
                             sharex="col", squeeze=False)

    for col, k in enumerate(selected):
        x, y = true[:, k], inferred[:, k]
        r = y - x
        ax = axes[0, col]
        ax.scatter(x, y, s=12, color="steelblue", alpha=0.4)
        ax.plot(bounds[k], bounds[k], "k--", lw=1.3)
        ax.set(xlim=bounds[k], ylim=bounds[k], ylabel="Inferred " + names[k],
               title=names[k] + " recovery")
        ax.grid(alpha=0.5)

        edges = np.linspace(*bounds[k], 9)
        centers = (edges[:-1] + edges[1:]) / 2
        means, q16, q84, ci_lo, ci_hi = [], [], [], [], []
        for i in range(8):
            mask = (x >= edges[i]) & (x < edges[i + 1] if i < 7 else x <= edges[i + 1])
            values = r[mask]
            means.append(values.mean())
            q16.append(np.percentile(values, 16))
            q84.append(np.percentile(values, 84))
            boot = rng.choice(values, (nboot, len(values)), replace=True).mean(axis=1)
            ci_lo.append(np.percentile(boot, 2.5))
            ci_hi.append(np.percentile(boot, 97.5))
        means = np.array(means)
        ax = axes[1, col]
        ax.scatter(x, r, s=10, color="steelblue", alpha=0.3,
                   label="Held-out realisations")
        ax.fill_between(centers, q16, q84, color="tab:green", alpha=0.4,
                        label="16–84% residual band")
        ax.plot(centers, means, "-o", color="steelblue", lw=2, ms=4,
                label="Mean residual")
        ax.errorbar(centers, means,
                    yerr=[means - np.array(ci_lo), np.array(ci_hi) - means],
                    fmt="none", color="black", capsize=3, lw=1,
                    label="95% bootstrap CI")
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set(xlim=bounds[k], xlabel="True " + names[k], ylabel=res_names[k])
        ax.grid(alpha=0.5)
        print(f"{names[k]}: Bias = {r.mean():.5f}, RMSE = {np.sqrt(np.mean(r**2)):.5f}")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(r"SBI using $C_\ell^{gg}+C_\ell^{vg}+C_\ell^{vv}$", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    return _finish(fig, _root(pipe) / "diagnostics" / "recovery.png", save)


# 03 / Official sbi SBC: numerical step, saved separately from plotting.
def compute_calibration(pipe, num_posterior_samples=1000):
    import torch
    from sbi.diagnostics import run_sbc, check_sbc

    root = _root(pipe)
    data = np.load(root / "heldout_data.npz")
    theta = torch.from_numpy(data["theta_test"]).float()
    x = torch.from_numpy(data["x_test"]).float()
    posterior = pipe.posterior
    if posterior is None:
        posterior = torch.load(root / "posterior.pt", map_location="cpu", weights_only=False)

    ranks, dap_samples = run_sbc(theta, x, posterior,
                                 num_posterior_samples=num_posterior_samples,
                                 use_batched_sampling=False)
    stats = check_sbc(ranks, theta, dap_samples,
                      num_posterior_samples=num_posterior_samples)
    dest = root / "calibration_sbc.npz"
    np.savez(dest, ranks=ranks.detach().cpu().numpy(),
             dap_samples=dap_samples.detach().cpu().numpy(),
             num_posterior_samples=num_posterior_samples,
             ks_pvals=np.asarray(stats["ks_pvals"]),
             c2st_ranks=np.asarray(stats["c2st_ranks"]),
             c2st_dap=np.asarray(stats["c2st_dap"]))
    print("Saved:", dest)
    return stats


def plot_calibration(pipe, test="both", save=False):
    import torch
    from scipy.stats import binom
    from sbi.analysis.plot import sbc_rank_plot

    data = np.load(_root(pipe) / "calibration_sbc.npz")
    ranks = torch.from_numpy(data["ranks"])
    nsamp = int(data["num_posterior_samples"])
    print("KS p-values:", data["ks_pvals"])
    print("C2ST ranks: ", data["c2st_ranks"])
    print("C2ST DAP:   ", data["c2st_dap"])
    figures = []

    if test in ("both", "sbc"):
        fig, _ = sbc_rank_plot(ranks, nsamp, plot_type="hist", num_bins=20,
                               parameter_labels=PARAMS, figsize=(12, 4))
        figures.append(_finish(fig, _root(pipe) / "diagnostics" / "sbc.png", save))

    if test in ("both", "coverage"):
        u = ranks.detach().cpu().numpy() / nsamp
        R = len(u)
        alphas = np.linspace(0.05, 0.95, 19)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for k, ax in enumerate(axes):
            coverage = np.array([
                np.mean((u[:, k] >= (1 - a) / 2) & (u[:, k] <= (1 + a) / 2))
                for a in alphas
            ])
            lower = binom.ppf(0.025, R, alphas) / R
            upper = binom.ppf(0.975, R, alphas) / R
            ax.fill_between(alphas, lower, upper, color="0.85",
                            label="95% binomial reference")
            ax.plot([0, 1], [0, 1], "--", color="0.35", label="Ideal calibration")
            ax.plot(alphas, coverage, "-o", color=["#2b6c9e", "#d97706"][k],
                    lw=2, ms=4, label="Empirical coverage")
            ax.set(xlim=(0, 1), ylim=(0, 1),
                   xlabel=r"Nominal credibility $\alpha$",
                   ylabel=r"Empirical coverage $\hat P_\alpha$",
                   title=f"Coverage: {PARAMS[k]}")
            ax.grid(alpha=0.5)
            ax.legend(frameon=False)
        plt.tight_layout()
        figures.append(_finish(fig, _root(pipe) / "diagnostics" / "coverage.png", save))
    return figures


# 04 / Both stability experiments from stability.py: plotting only.
def plot_stability(pipe, test="both", save=False):
    root = _root(pipe) / "stability"
    tests = {
        "seeds": ("training_seeds/summary.npz", "Training-seed stability",
                  "Training seed", "tab:blue"),
        "heldout": ("heldout_rotation/summary.npz", "Held-out rotation stability",
                    "Held-out fold", "tab:green"),
    }
    selected = tests if test == "both" else {test: tests[test]}
    figures = []

    for name, (filename, title, xlabel, color) in selected.items():
        data = np.load(root / filename)
        rmse, bias = data["rmse"], data["bias"]
        x = data["seeds"] if name == "seeds" else np.arange(1, len(rmse) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for j in range(2):
            ax = axes[0, j]
            ax.plot(x, rmse[:, j], "o-", color=color)
            ax.axhline(rmse[:, j].mean(), color=color, ls="--")
            ax.set(title=PARAMS[j], ylabel="Held-out RMSE")
            ax.grid(alpha=0.3)

            ax = axes[1, j]
            ax.plot(x, bias[:, j], "o-", color=color)
            ax.axhline(bias[:, j].mean(), color=color, ls="--")
            ax.axhline(0, color="black", ls=":")
            ax.set(xlabel=xlabel, ylabel="Mean bias")
            ax.grid(alpha=0.3)
            sd = rmse[:, j].std(ddof=1) if len(rmse) > 1 else float("nan")
            print(f"{title}, {PARAMS[j]}: mean RMSE = {rmse[:, j].mean():.4f}; "
                  f"SD across runs = {sd:.4f}")
        fig.suptitle(title)
        plt.tight_layout()
        figures.append(_finish(fig, root / f"{name}_stability.png", save))
    return figures


# 05 / Knox: the full 136-spectrum test from the source notebook.
# Explicit numerical stage: never run automatically from plot_knox().
def compute_knox(pipe, nsim=2000, nboot=1500, fNL=0.0, bv=0.5):
    import healpy as hp
    from .theory import load_theory, make_theory_cls

    cfg = pipe.settings
    cls = load_theory(cfg)
    gg, vg, vv = make_theory_cls(fNL, bv, cls, cap=cfg.cap)
    n, L = cfg.nbins, cfg.lmax
    ell = np.arange(2, L + 1)
    nf = 2 * n
    S = np.zeros((nf, nf, L + 1))
    S[:n, :n] = gg
    S[n:, n:] = vv
    S[n:, :n] = vg
    S[:n, n:] = vg.transpose(1, 0, 2)
    pairs_hp = ([(i, i) for i in range(nf)]
                + [(i, i + d) for d in range(1, nf) for i in range(nf - d)])
    cls_hp = [S[i, j] for i, j in pairs_hp]
    group_hp = np.array([0 if j < n else (1 if i < n else 2) for i, j in pairs_hp])
    order = np.argsort(group_hp, kind="stable")
    pairs = np.array(pairs_hp)[order]
    groups = group_hp[order]
    a, b = pairs.T
    npair = len(pairs)

    draws = np.empty((nsim, npair, len(ell)))
    for r in range(nsim):
        np.random.seed(2000000 + r)
        alms = hp.synalm(cls_hp, lmax=L, new=True)
        draws[r] = np.asarray(hp.alm2cl(alms))[order, 2:]

    V_knox = (S[a, a, 2:] * S[b, b, 2:] + S[a, b, 2:]**2) / (2 * ell + 1)
    if np.any(V_knox <= 0):
        raise ValueError("Non-positive Knox variance detected.")
    R = draws.var(axis=0, ddof=1) / V_knox

    rng = np.random.default_rng(123)
    R_boot = np.empty((nboot, npair, len(ell)))
    for k in range(nboot):
        idx = rng.integers(0, nsim, nsim)
        R_boot[k] = draws[idx].var(axis=0, ddof=1) / V_knox
    lo, hi = np.percentile(R_boot, [2.5, 97.5], axis=0)
    outside = (lo > 1) | (hi < 1)
    fraction_outside = outside.mean(axis=0)

    dest = _root(pipe) / "knox_validation.npz"
    np.savez(dest, R=R, lo=lo, hi=hi, ell=ell, groups=groups,
             fraction_outside=fraction_outside, pairs=pairs, nsim=nsim, fNL=fNL, bv=bv)
    print("Saved:", dest)
    return dest


def plot_knox(pipe, test="both", save=False):
    from matplotlib.colors import TwoSlopeNorm

    data = np.load(_root(pipe) / "knox_validation.npz")
    R, lo, hi, ell, groups = (data[k] for k in ("R", "lo", "hi", "ell", "groups"))
    fraction_outside = data["fraction_outside"]
    nsim, fNL, bv = int(data["nsim"]), float(data["fNL"]), float(data["bv"])
    n = pipe.settings.nbins
    figs = []

    if test in ("both", "full"):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]},
                                      layout="constrained")
        im = ax.imshow(R, origin="lower", aspect="auto", interpolation="nearest",
                       extent=[1.5, ell[-1] + 0.5, -0.5, len(R) - 0.5],
                       cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.65, vcenter=1, vmax=1.35))
        ngg, nvg = n * (n + 1) // 2, n * n
        ax.axhline(ngg - 0.5, color="black", lw=1.5)
        ax.axhline(ngg + nvg - 0.5, color="black", lw=1.5)
        ax.set_yticks([ngg / 2 - 0.5, ngg + nvg / 2 - 0.5,
                       ngg + nvg + ngg / 2 - 0.5],
                      [rf"$gg$ ({ngg})", rf"$vg$ ({nvg})", rf"$vv$ ({ngg})"])
        ax.set_ylabel(f"Spectrum pairs ({len(R)} total)")
        ax.set_title(f"Knox validation: {nsim} independent skies, "
                     rf"$f_{{\rm NL}}={fNL:g}$, $b_v={bv:g}$")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r"$R_\ell^{ab} = \mathrm{Var}_{sim}/\mathrm{Var}_{Knox}$")
        ax2.plot(ell, fraction_outside, "o-", ms=3)
        ax2.axhline(0.05, color="black", ls="--", label="5% pointwise reference")
        ax2.set(xlabel=r"Multipole $\ell$", ylabel="Fraction outside\n95% CI",
                xlim=(2, ell[-1]),
                ylim=(0, max(0.15, fraction_outside.max() * 1.15)))
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.25)
        figs.append(_finish(fig, _root(pipe) / "diagnostics" / "knox_full.png", save))

    if test in ("both", "summary"):
        fig, ax = plt.subplots(figsize=(9, 5))
        points = np.unique(np.r_[np.arange(0, len(ell), 4), len(ell) - 1])
        # The notebook's three representative spectra: g0g0, v0g0, v0v0.
        pairs = data["pairs"]
        for pair, label in [((0, 0), "gg"), ((0, n), "vg"), ((n, n), "vv")]:
            idx = np.flatnonzero(np.all(pairs == pair, axis=1))[0]
            ax.errorbar(ell[points], R[idx, points],
                        yerr=[R[idx, points] - lo[idx, points],
                              hi[idx, points] - R[idx, points]],
                        fmt="o-", capsize=3, label=label)
        ax.axhline(1, color="black", linestyle="--", label="Knox prediction")
        ax.set(xlabel=r"Multipole $\ell$",
               ylabel=r"$\mathrm{Var}_{sim}/\mathrm{Var}_{Knox}$",
               title=f"Knox validation: {nsim} independent skies")
        ax.legend()
        ax.grid(alpha=0.5)
        plt.tight_layout()
        figs.append(_finish(fig, _root(pipe) / "diagnostics" / "knox_summary.png", save))

    print("Mean variance ratio over all multipoles:")
    for k, name in enumerate(["gg", "vg", "vv"]):
        mask = groups == k
        outside = (lo[mask] > 1) | (hi[mask] < 1)
        print(f"{name}: mean R = {R[mask].mean():.4f}, "
              f"outside 95% CI = {outside.mean():.1%}")
    print(f"All spectra: mean R = {R.mean():.4f}")
    return figs


# One dispatcher for the future Pipeline.result() method.
def result(pipe, *names, save=False, **options):
    plots = {
        "posterior": plot_posterior,
        "recovery": plot_recovery,
        "calibration": plot_calibration,
        "stability": plot_stability,
        "fisher": plot_fisher,
        "knox": plot_knox,
    }
    if not names:
        names = tuple(name for name in plots if name != "posterior")
    if options and len(names) != 1:
        raise ValueError("Options such as test= or parameter= require one diagnostic.")

    outputs = {}
    for name in names:
        if name not in plots:
            raise ValueError(f"Unknown diagnostic: {name}. Choose from {list(plots)}")
        try:
            outputs[name] = plots[name](pipe, save=save, **options)
        except FileNotFoundError:
            if len(names) == 1:
                raise
            print(f"Skipping {name}: saved numerical results not found.")
    return outputs
