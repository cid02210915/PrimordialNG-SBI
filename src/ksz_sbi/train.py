from pathlib import Path
import torch
from sbi.inference import NPE
from .models import build_model

def train_posterior(theta_train, x_train, settings):

    # 1. Build prior and neural posterior architecture
    prior, neural_posterior = build_model(settings)

    # 2. Initialise NPE
    inference = NPE(
        prior=prior,
        density_estimator=neural_posterior
    )

    # 3. Provide training simulations and train
    density_estimator = (
        inference
        .append_simulations(theta_train, x_train)
        .train()
    )

    # 4. Build trained posterior
    posterior = inference.build_posterior(
        density_estimator
    )

    # 5. Save trained posterior
    path = Path(settings.data_root) / "posterior.pt"
    torch.save(posterior, path)
    print("Saved trained posterior:", path)

    return posterior
