import numpy as np
import torch
from pathlib import Path

from .train import train_posterior
from .inference import sample_posterior
from .dataset import prepare_data

# ============================================================
# TEST 1 — TRAINING-SEED STABILITY
# Same training data, same held-out data, different NPE seeds
# ============================================================
def training_seed_test(pipe, seeds=(1, 2, 3, 4, 5)):

    if pipe.x_train is None or pipe.x_test is None:
        raise RuntimeError("Call pipe.prepare_data() first.")

    root = Path(pipe.settings.data_root) / "stability" / "training_seeds"
    root.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in seeds]
    rmse = np.zeros((len(seeds), 2))
    bias = np.zeros((len(seeds), 2))
    true = pipe.theta_test.detach().cpu().numpy()

    for k, seed in enumerate(seeds):

        folder = root / f"seed_{seed}"
        folder.mkdir(exist_ok=True)  # Prevent overwriting previous runs

        changes = {"data_root": str(folder)}
        if hasattr(pipe.settings, "training_seed"):
            changes["training_seed"] = seed
        cfg = pipe.settings.updated(**changes)

        # Different random training initialisation / trajectory
        torch.manual_seed(seed)

        posterior = train_posterior(
            pipe.theta_train,
            pipe.x_train,
            cfg
        )
        if not (folder / "posterior.pt").exists():
            torch.save(posterior, folder / "posterior.pt")

        # Keep posterior-sampling randomness reproducible
        torch.manual_seed(2026)

        result = sample_posterior(
            posterior,
            pipe.x_test,
            n_samples=cfg.resolved_posterior_samples,
            label=f"Training seed {seed}"
        )

        error = result["median"] - true

        rmse[k] = np.sqrt(np.mean(error**2, axis=0))
        bias[k] = np.mean(error, axis=0)

        np.savez(
            folder / "results.npz",
            seed=seed,
            theta_true=true,
            median=result["median"],
            rmse=rmse[k],
            bias=bias[k]
        )
    
        np.savez(
            root / "summary.npz",
            seeds=seeds,
            rmse=rmse,
            bias=bias
        )
    return rmse, bias


# ============================================================
# TEST 2 — HELD-OUT ROTATION STABILITY
# Rotate held-out skies; retrain MAF on each remaining training set
# ============================================================

def heldout_rotation_test(pipe, n_folds=16, split_seed=12345, train_seed=42):
    
    from .dataset import prepare_data
    settings = pipe.settings

    # Load all original simulations without filtering
    raw_cfg = settings.updated(
        data_mode="existing",
        remove_flagged_realisations=False
    )
    theta_train, x_train, theta_test, x_test = prepare_data(raw_cfg)

    # Combine training and held-out simulations, preserving (x, theta) pairs
    x = torch.cat((x_train, x_test), dim=0)
    theta = torch.cat((theta_train, theta_test), dim=0)

    # Randomise simulation order once, reproducibly
    nheld = len(x_test)
    indices = np.random.default_rng(split_seed).permutation(len(x))

    # Ensure held-out folds do not overlap
    if not 1 <= n_folds <= len(x) // nheld:
        raise ValueError("Too many folds for non-overlapping held-out sets.")

    # Store stability results separately from the original pipeline
    root = Path(settings.data_root) / "stability" / "heldout_rotation"
    root.mkdir(parents=True, exist_ok=True)

    if (root / "summary.npz").exists():
        raise FileExistsError("Rotation results already exist.")

    cfg = settings.updated(data_root=str(root))

    # Store RMSE and bias for f_NL and b_v in each fold
    rmse = np.zeros((n_folds, 2))
    bias = np.zeros((n_folds, 2))

    for k in range(n_folds):

        # Rotate 2,000 held-out skies; train on all remaining skies
        start = k * nheld
        end = start + nheld

        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        # Recalculate 10×IQR thresholds using training skies only
        if settings.remove_flagged_realisations:
            X = x[train_idx].numpy()
            q1, q3 = np.percentile(X, [25, 75], axis=0)
            iqr = q3 - q1

            # Remove flagged training skies; keep held-out skies untouched
            keep = ~np.any(
                (X < q1 - 10 * iqr) | (X > q3 + 10 * iqr),
                axis=1
            )
            train_idx = train_idx[keep]

        # Retrain MAF from scratch with the same random seed
        torch.manual_seed(train_seed)

        posterior = train_posterior(
            theta[train_idx],
            x[train_idx],
            cfg
        )
        # Fix posterior-sampling randomness across folds
        torch.manual_seed(2026)

        # Infer parameters for this fold's unseen held-out skies
        result = sample_posterior(
            posterior,
            x[test_idx],
            n_samples=cfg.resolved_posterior_samples,
            label=f"Fold {k+1}"
        )
        # Compare posterior medians with true parameters
        true = theta[test_idx].detach().cpu().numpy()
        error = result["median"] - true

        # Compute RMSE and mean bias for both parameters
        rmse[k] = np.sqrt(np.mean(error**2, axis=0))
        bias[k] = np.mean(error, axis=0)

        # Save results after each completed fold
        np.savez(
            root / "summary.npz",
            rmse=rmse[:k+1],
            bias=bias[:k+1]
        )
        # Delete temporary MAF checkpoint to save storage
        (root / "posterior.pt").unlink(missing_ok=True)

        print(f"Fold {k+1}/{n_folds} complete")

    return rmse, bias
