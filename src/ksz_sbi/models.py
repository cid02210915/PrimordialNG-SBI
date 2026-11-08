import torch
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding


def build_model(settings):

    # 1. Prior
    prior = BoxUniform(
        low=torch.tensor(
            [settings.fNL_prior[0], settings.bv_prior[0]],
            dtype=torch.float32
        ),
        high=torch.tensor(
            [settings.fNL_prior[1], settings.bv_prior[1]],
            dtype=torch.float32
        )
    )

    # 2. Optional FCEmbedding
    embedding_net = torch.nn.Identity()

    if settings.resolved_embedding == "fc":
        dims = settings.resolved_embedding_dimensions

        embedding_net = FCEmbedding(
            input_dim=settings.expected_feature_dimension,
            output_dim=dims["output_dim"],
            num_layers=dims["layers"],
            num_hiddens=dims["hidden_dim"]
        )

    # 3. Posterior density model
    neural_posterior = posterior_nn(
        model=settings.model,
        embedding_net=embedding_net,
        z_score_x=settings.z_score_x
    )

    return prior, neural_posterior
