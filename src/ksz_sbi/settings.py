"""Configuration settings for the kSZ SBI pipeline."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Settings:
    # Input representation: 'cl', 'map', or 'alm' ('a_lm' is also accepted).
    input_type: str = "cl"
    spectra: tuple[str, ...] = ("gg", "vg", "vv")  # used for C_l input

    # The original physics/simulation baseline.
    nbins: int = 8
    nside: int = 32
    lmax: int = 32
    ell_min: int = 2
    cap: str = "sgc"

    # Physical-parameter priors, with bounds in (lower, upper) order.
    fNL_prior: tuple[float, float] = (-100.0, 100.0)
    bv_prior: tuple[float, float] = (0.20, 0.80)

    # Dataset size and seeds from the original notebook.
    n_train_parameters: int = 2000
    realisations_per_parameter: int = 4
    n_heldout_parameters: int = 1000
    train_parameter_seed: int = 12345
    heldout_parameter_seed: int = 987654
    train_sky_seed_base: int = 1_000_000
    heldout_sky_seed_base: int = 10_000_000
    remove_flagged_realisations: bool = True
    data_mode: str = "existing"  # "existing" or "generate"

    # SBI choices.
    model: str = "maf"  # 'maf' or 'nsf'
    embedding: str = "auto"  # 'auto', 'none', 'fc'
    embedding_output_dim: int | None = None
    embedding_hidden_dim: int | None = None
    embedding_layers: int | None = None
    z_score_x: str = "independent"
    posterior_samples: int | None = None

    # Data and theory file locations.
    data_root: str = "/data/xl691/kSZCeell_sbi"
    theory_root: str = (
        "/data/xl691/kSZCeell/"
        "theory_spectra/b26e1d93ffe8d1979819915366c99ab3/Cls"
    )
    overwrite: bool = False

    def __post_init__(self) -> None:
        # Accept the names already used in the user's notebooks/conversation.
        aliases = {"field": "map", "a_lm": "alm", "alms": "alm"}
        name = self.input_type.lower()
        object.__setattr__(self, "input_type", aliases.get(name, name))
        object.__setattr__(self, "model", self.model.lower())
        object.__setattr__(self, "embedding", self.embedding.lower())
        object.__setattr__(self, "spectra", tuple(s.lower() for s in self.spectra))
        object.__setattr__(self, "fNL_prior", tuple(self.fNL_prior))
        object.__setattr__(self, "bv_prior", tuple(self.bv_prior))

        if self.input_type not in {"cl", "map", "alm"}:
            raise ValueError("input_type must be 'cl', 'map', or 'alm' (alias: 'a_lm').")
        if self.model not in {"maf", "nsf"}:
            raise ValueError("model must be 'maf' or 'nsf'.")
        if self.embedding not in {"auto", "none", "fc"}:
            raise ValueError("embedding must be 'auto', 'none', or 'fc'.")
        if self.z_score_x not in {"independent", "structured", "none"}:
            raise ValueError("z_score_x must be 'independent', 'structured', or 'none'.")
        if self.data_mode not in {"existing", "generate"}:
            raise ValueError("data_mode must be 'existing' or 'generate'.")
        if self.cap not in {"sgc", "ngc", "no_noise"}:
            raise ValueError("cap must be 'sgc', 'ngc', or 'no_noise' (supported by the original theory code).")
        if len(self.spectra) == 0 or len(set(self.spectra)) != len(self.spectra):
            raise ValueError("spectra must contain at least one distinct spectrum.")
        if set(self.spectra) - {"gg", "vg", "vv"}:
            raise ValueError("Supported spectra are 'gg', 'vg' and 'vv'.")
        if self.nbins < 1 or self.nside < 1 or self.nside & (self.nside - 1):
            raise ValueError("nbins must be positive and nside must be a power of two.")
        if not (0 <= self.ell_min <= self.lmax):
            raise ValueError("Require 0 <= ell_min <= lmax.")
        for name, pair in (("fNL_prior", self.fNL_prior), ("bv_prior", self.bv_prior)):
            if len(pair) != 2 or not pair[0] < pair[1]:
                raise ValueError(f"{name} must be (lower_bound, upper_bound).")
        for name in (
            "n_train_parameters", "realisations_per_parameter", "n_heldout_parameters",
            "train_parameter_seed", "heldout_parameter_seed", "train_sky_seed_base",
            "heldout_sky_seed_base",
        ):
            value = getattr(self, name)
            if (value < 0 if "seed" in name else value <= 0):
                raise ValueError(f"{name} must be {'nonnegative' if 'seed' in name else 'positive'}.")
        for name in ("embedding_output_dim", "embedding_hidden_dim", "embedding_layers", "posterior_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive if specified.")

    @property
    def resolved_embedding(self) -> str:
        if self.embedding == "auto":
            # Reflects the original notebook: direct C_l, FC for map and a_lm.
            return "none" if self.input_type == "cl" else "fc"
        return self.embedding

    @property
    def resolved_embedding_dimensions(self) -> dict[str, int] | None:
        if self.resolved_embedding != "fc":
            return None
        # Preserves the notebook's choices.
        if self.input_type == "map":
            original = (8, 20, 2)
        elif self.input_type == "alm":
            original = (32, 128, 4)
        else:
            if self.embedding_output_dim is None or self.embedding_hidden_dim is None or self.embedding_layers is None:
                raise ValueError("FC embedding for C_l needs explicit output_dim, hidden_dim and layers.")
            original = (self.embedding_output_dim, self.embedding_hidden_dim, self.embedding_layers)
        return {
            "output_dim": self.embedding_output_dim or original[0],
            "hidden_dim": self.embedding_hidden_dim or original[1],
            "layers": self.embedding_layers or original[2],
        }

    @property
    def resolved_posterior_samples(self) -> int:
        # Original notebooks: 5,000 (C_l/map), 2,000 (a_lm).
        return self.posterior_samples or (2000 if self.input_type == "alm" else 5000)

    @property
    def expected_feature_dimension(self) -> int:
        if self.input_type == "cl":
            return len(self.spectra) * self.nbins**2 * (self.lmax - self.ell_min + 1)
        if self.input_type == "map":
            return 2 * self.nbins * 12 * self.nside**2
        # Healpy stores m >= 0; the original code retains real and imaginary parts.
        n_alm = ((self.lmax + 1) * (self.lmax + 2) - self.ell_min * (self.ell_min + 1)) // 2
        return 2 * self.nbins * 2 * n_alm

    def updated(self, **changes: Any) -> "Settings":
        """Make another validated configuration without altering this one."""
        return replace(self, **changes)