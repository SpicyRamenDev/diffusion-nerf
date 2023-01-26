import torch
import torch.nn as nn

from .utils import get_sph_harm_mat, compute_integrated_dir_enc


class IDEEmbedder(nn.Module):
    """PyTorch implementation of spherical harmonics embedding.
    """

    def __init__(self, degree, use_kappa=False, include_input=True):
        """Initialize the module.

        Args:
            degree (int): degree of the polynomials.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.

        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.degree = degree
        self.use_kappa = use_kappa
        self.include_input = include_input

        ml_array, mat = get_sph_harm_mat(deg_view=degree)
        self.ml_array = nn.Parameter(torch.tensor(ml_array), requires_grad=False)
        self.mat = nn.Parameter(mat, requires_grad=False)

        self.out_dim = 0
        if include_input:
            self.out_dim += 3
        if use_kappa:
            self.out_dim += 1
        self.out_dim += 2 * self.ml_array.shape[1]

    def forward(self, dirs, kappa_inv=None):
        """Embeds the direction.

        Args:
            dirs (torch.FloatTensor): Coordinates of shape [N, 3]

            kappa (torch.Float): kappa or shape [N, 1]

        Returns:
            (torch.FloatTensor): Embeddings of shape [N, 4 + out_dim] or [N, 3 + out_dim] or [N, out_dim].
        """
        if kappa_inv is None:
            kappa_inv = torch.zeros_like(dirs[..., :1])

        output = compute_integrated_dir_enc(self.ml_array, self.mat, dirs, kappa_inv)

        if self.include_input:
            output = torch.cat([output, dirs], axis=-1)
        if self.use_kappa:
            output = torch.cat([output, kappa_inv], axis=-1)

        return output


def get_ide_embedder(degree, use_kappa):
    """Utility function to get a spherical harmonics encoding embedding.

    Args:
        degree (int): The number of frequencies used to define the PE:
            [2^0, 2^1, 2^2, ... 2^(frequencies - 1)].
        use_kappa (bool): whether to use kappa or not.
        active (bool): If false, will return the identity function.

    Returns:
        (nn.Module, int):
        - The embedding module
        - The output dimension of the embedding.
    """
    encoder = IDEEmbedder(degree, use_kappa)
    return encoder, encoder.out_dim
