
"""Main controller for the kSZ SBI pipeline."""

from __future__ import annotations
from dataclasses import replace
from typing import Any
from .settings import Settings

class Pipeline:

    def __init__(self, settings: Settings | None = None, **choices: Any):

        # Allow input="cl", "map", or "a_lm"
        if "input" in choices:
            if "input_type" in choices:
                raise TypeError("Use 'input' OR 'input_type', not both.")
            choices["input_type"] = choices.pop("input")

        if settings is not None and not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance or None.")

        self.settings = (
            Settings(**choices)
            if settings is None
            else replace(settings, **choices)
        )

    def with_changes(self, **changes: Any) -> "Pipeline":
        """Create a new pipeline with updated settings."""
        return Pipeline(settings=self.settings, **changes)


    def prepare_data(self):
        from .dataset import prepare_data
        (
            self.theta_train,
            self.x_train,
            self.theta_test,
            self.x_test,
        ) = prepare_data(self.settings)

    def train(self):
        from .train import train_posterior

        self.posterior = train_posterior(
            self.theta_train,
            self.x_train,
            self.settings
        )
        return self.posterior


    def infer(self):
        from .inference import sample_posterior
        cfg = self.settings

        self.inference_results = sample_posterior(
            posterior=self.posterior,
            x=self.x_test,
            theta_true=self.theta_test,
            n_samples=cfg.resolved_posterior_samples,
            save_path=(
                f"{cfg.data_root}/"
                f"heldout_{len(self.x_test)}_sbi_"
                f"{cfg.input_type}_{cfg.resolved_embedding}_"
                f"{cfg.model}_results.npz"
            )
        )
        return self.inference_results

    def result(self, *names, **kwargs):
        from .diagnostics import result
        return result(self, *names, **kwargs)

    def run(self):
        self.prepare_data()
        self.train()
        return self.infer()
