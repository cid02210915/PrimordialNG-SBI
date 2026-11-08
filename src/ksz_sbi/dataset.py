from pathlib import Path
import numpy as np
import torch

from .theory import load_theory, make_theory_cls
from .simulator import simulate_sky
from .features import extract_features

def _generate_saved_data(settings, root):

    ntrain = settings.n_train_parameters
    nreal = settings.realisations_per_parameter
    nheld = settings.n_heldout_parameters

    train_dir = root / f"train_{ntrain}x{nreal}_nside{settings.nside}_lmax{settings.lmax}"
    theory_dir = root / f"theory_{ntrain}_lmax{settings.lmax}"
    held_dir = root / f"heldout_theory_{nheld}_lmax{settings.lmax}"

    if not settings.overwrite and (
        any(train_dir.glob("train_p*_r*.npz"))
        or (held_dir / f"parameters_heldout_{nheld}.npz").exists()
    ):
        raise FileExistsError(
            "Saved data already exist. Use data_mode='existing' "
            "or choose a new data_root."
        )

    for path in (train_dir, theory_dir, held_dir):
        path.mkdir(parents=True, exist_ok=True)

    cls = load_theory(settings)

    # -------------------------------------------
    # Training parameters
    # -------------------------------------------
    rng = np.random.default_rng(settings.train_parameter_seed)

    fNL_all = rng.uniform(
        *settings.fNL_prior, ntrain
    )
    bv_all = rng.uniform(
        *settings.bv_prior, ntrain
    )

    shape = (
        ntrain, settings.nbins, settings.nbins,
        settings.lmax + 1
    )
    train_banks = {
        name: np.empty(shape)
        for name in ("gg", "vg", "vv")
    }

    for p in range(ntrain):
        gg, vg, vv = make_theory_cls(
            fNL_all[p], bv_all[p], cls, cap=settings.cap
        )

        for name, cl in zip(("gg", "vg", "vv"), (gg, vg, vv)):
            train_banks[name][p] = cl

        for r in range(nreal):
            seed = settings.train_sky_seed_base + p * nreal + r
            sim = simulate_sky(gg, vg, vv, settings, seed)

            np.savez(
                train_dir / f"train_p{p:04d}_r{r}.npz",
                parameter_id=p,
                realisation_id=r,
                seed=seed,
                fNL=fNL_all[p],
                bv=bv_all[p],
                maps_g=sim["maps_g"],
                maps_v=sim["maps_v"],
                Cl_gg=sim["Cl_gg"],
                Cl_vg=sim["Cl_vg"],
                Cl_vv=sim["Cl_vv"]
            )

    np.savez(
        theory_dir / f"parameters_{ntrain}.npz",
        parameter_id=np.arange(ntrain),
        fNL=fNL_all,
        bv=bv_all
    )

    for name, bank in train_banks.items():
        np.savez(
            theory_dir / f"Cl_{name}_{ntrain}.npz",
            ells=np.arange(settings.lmax + 1),
            Cl=bank
        )

    # -------------------------------------------
    # Independent held-out parameters
    # -------------------------------------------
    rng_held = np.random.default_rng(
        settings.heldout_parameter_seed
    )

    fNL_held = rng_held.uniform(
        *settings.fNL_prior, nheld
    )
    bv_held = rng_held.uniform(
        *settings.bv_prior, nheld
    )

    held_shape = (
        nheld, settings.nbins, settings.nbins,
        settings.lmax + 1
    )
    held_banks = {
        name: np.empty(held_shape)
        for name in ("gg", "vg", "vv")
    }

    for h in range(nheld):
        gg, vg, vv = make_theory_cls(
            fNL_held[h], bv_held[h], cls, cap=settings.cap
        )

        for name, cl in zip(("gg", "vg", "vv"), (gg, vg, vv)):
            held_banks[name][h] = cl

    np.savez(
        held_dir / f"parameters_heldout_{nheld}.npz",
        parameter_id=np.arange(nheld),
        fNL=fNL_held,
        bv=bv_held
    )

    for name, bank in held_banks.items():
        np.savez(
            held_dir / f"Cl_{name}_heldout_{nheld}.npz",
            ells=np.arange(settings.lmax + 1),
            Cl=bank
        )

def prepare_data(settings):

    root = Path(settings.data_root)

    ntrain = settings.n_train_parameters
    nreal = settings.realisations_per_parameter
    nheld = settings.n_heldout_parameters

    if settings.data_mode == "generate":
        _generate_saved_data(settings, root)

    # -------------------------------------------
    # 1. Load training realisations
    # -------------------------------------------
    train_dir = root / f"train_{ntrain}x{nreal}_nside{settings.nside}_lmax{settings.lmax}"
    held_dir = root / f"heldout_theory_{nheld}_lmax{settings.lmax}"

    files = sorted(train_dir.glob("train_p*_r*.npz"))

    if len(files) != ntrain * nreal:
        raise FileNotFoundError(
            f"Expected {ntrain * nreal} training files "
            f"in {train_dir}; found {len(files)}."
        )

    # -------------------------------------------
    # 2. Optional automatic 10xIQR filtering
    # -------------------------------------------
    if settings.remove_flagged_realisations:
        
        ref_settings = settings.updated(
            input_type="cl",
            spectra=("gg", "vg", "vv"),
            ell_min=2
        )
        X_ref = np.empty(
            (len(files), ref_settings.expected_feature_dimension),
            dtype=np.float32
        )
        for i, path in enumerate(files):
            with np.load(path) as sim:
                X_ref[i] = extract_features(sim, ref_settings)

        q1, q3 = np.percentile(X_ref, [25, 75], axis=0)
        iqr = q3 - q1

        flagged = np.any(
            (X_ref < q1 - 10.0 * iqr)
            | (X_ref > q3 + 10.0 * iqr),
            axis=1
        )
        print(
            "10xIQR flagged training realisations:",
            int(flagged.sum())
        )
        files = [
            path for i, path in enumerate(files)
            if not flagged[i]
        ]

    # -------------------------------------------
    # 3. Extract selected training input
    # -------------------------------------------
    x_train = np.empty(
        (len(files), settings.expected_feature_dimension),
        dtype=np.float32
    )
    theta_train = np.empty(
        (len(files), 2), dtype=np.float32
    )

    for i, path in enumerate(files):
        with np.load(path) as sim:
            x_train[i] = extract_features(sim, settings)
            theta_train[i] = (
                float(sim["fNL"]), float(sim["bv"])
            )

    # -------------------------------------------
    # 4. Load held-out theory
    # -------------------------------------------
    with np.load(
        held_dir / f"parameters_heldout_{nheld}.npz"
    ) as pars:
        fNL_held = pars["fNL"]
        bv_held = pars["bv"]

    held_banks = {}

    for name in ("gg", "vg", "vv"):
        with np.load(
            held_dir / f"Cl_{name}_heldout_{nheld}.npz"
        ) as d:
            held_banks[name] = d["Cl"]

    if (
        len(fNL_held) != nheld
        or any(len(bank) != nheld for bank in held_banks.values())
    ):
        raise ValueError("Held-out theory-bank size does not match settings.")

    # -------------------------------------------
    # 5. Generate held-out realisations
    # -------------------------------------------
    x_test = np.empty(
        (nheld, settings.expected_feature_dimension),
        dtype=np.float32
    )

    theta_test = np.column_stack(
        (fNL_held, bv_held)
    ).astype(np.float32)

    for h in range(nheld):
        seed = settings.heldout_sky_seed_base + h

        sim = simulate_sky(
            held_banks["gg"][h],
            held_banks["vg"][h],
            held_banks["vv"][h],
            settings,
            seed
        )
        x_test[h] = extract_features(sim, settings)

    # Save held-out inputs and true parameters
    np.savez(
        root / "heldout_data.npz",
        x_test=x_test,
        theta_test=theta_test
    )
    
    return (
        torch.from_numpy(theta_train),
        torch.from_numpy(x_train),
        torch.from_numpy(theta_test),
        torch.from_numpy(x_test)
    )
