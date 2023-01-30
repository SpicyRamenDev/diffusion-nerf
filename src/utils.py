# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for reflection directions and directional encodings."""

import torch
import numpy as np

from kaolin.render.camera import Camera
from wisp.ops.raygen.raygen import generate_default_grid, generate_centered_pixel_coords, generate_pinhole_rays


@torch.cuda.amp.autocast(enabled=False)
def rays_aabb_bounds(rays, aabb=None):
    if aabb is None:
        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], device=rays.origins.device)

    tmin = (aabb[0] - rays.origins) / rays.dirs
    tmax = (aabb[1] - rays.origins) / rays.dirs
    tmin, tmax = torch.min(tmin, tmax), torch.max(tmin, tmax)
    tmin, tmax = tmin.max(dim=-1, keepdim=True)[0], tmax.min(dim=-1, keepdim=True)[0]

    mask = (tmax < tmin).reshape(-1)
    tmin[mask] = rays.dist_max
    tmax[mask] = rays.dist_max

    if mask.any():
        print(mask.sum().item())
        print((rays.origins[~mask]+tmax[~mask]*rays.dirs[~mask]).abs().flatten().max(dim=-1)[0])

    tmin = tmin.clamp(min=rays.dist_min, max=rays.dist_max)
    tmax = tmax.clamp(min=rays.dist_min, max=rays.dist_max)

    return tmin, tmax, ~mask


def generate_camera_rays(camera):
    ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                              camera.width, camera.height,
                                              device='cuda')
    return generate_pinhole_rays(camera, ray_grid)


def sample(value_range=(0, 1)):
    return value_range[0] + (value_range[1] - value_range[0]) * torch.rand(1)


def spherical_to_cartesian(azimuth, polar, distance=1.0):
    azimuth = azimuth * torch.pi / 180.0
    polar = polar * torch.pi / 180.0
    x = distance * torch.sin(polar) * torch.cos(azimuth)
    y = distance * torch.cos(polar)
    z = -distance * torch.sin(polar) * torch.sin(azimuth)
    return torch.tensor([x, y, z])


def get_rotation_matrix(azimuth, polar):
    azimuth = azimuth * torch.pi / 180.0
    polar = polar * torch.pi / 180.0
    cos_azimuth, cos_polar = torch.cos(azimuth), torch.cos(polar)
    sin_azimuth, sin_polar = torch.sin(azimuth), torch.sin(polar)
    return torch.tensor([
        [cos_polar * cos_azimuth, sin_polar * cos_azimuth, sin_azimuth],
        [-sin_polar, cos_polar, 0.0],
        [-cos_polar * sin_azimuth, -sin_polar * sin_azimuth, cos_azimuth]
    ])


def sample_polar(polar_range=(0, 100), uniform=True):
    if not uniform:
        return sample(polar_range)
    max_polar = min(polar_range[1] * torch.pi / 180.0, torch.pi - torch.finfo(torch.float32).eps)
    min_polar = max(polar_range[0] * torch.pi / 180.0, torch.finfo(torch.float32).eps)
    cos_max, cos_min = torch.cos(torch.tensor([min_polar, max_polar]))
    polar = torch.acos(sample((cos_min, cos_max))) * 180.0 / torch.pi
    return polar


def sample_spherical_uniform(azimuth_range=(0, 360), polar_range=(0, 100), uniform=True):
    azimuth = sample(azimuth_range)
    polar = sample_polar(polar_range, uniform)

    coords = spherical_to_cartesian(azimuth, polar)
    return coords, azimuth, polar


def reflect(viewdirs, normals):
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Args:
      viewdirs: [..., 3] array of view directions.
      normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
      [..., 3] array of reflection directions.
    """
    return 2.0 * torch.sum(
        normals * viewdirs, axis=-1, keepdim=True) * normals - viewdirs


def l2_normalize(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(torch.clamp(torch.sum(x ** 2, axis=-1, keepdim=True), min=eps))


def compute_weighted_mae(weights, normals, normals_gt):
    """Compute weighted mean angular error, assuming normals are unit length."""
    one_eps = 1 - torch.finfo(torch.float32).eps
    return (weights * torch.arccos(
        torch.clip((normals * normals_gt).sum(-1), -one_eps,
                   one_eps))).sum() / weights.sum() * 180.0 / torch.pi


def compute_mean_angular_error(normals, normals_gt):
    """Compute mean angular error, assuming normals are unit length."""
    one_eps = 1 - torch.finfo(torch.float32).eps
    dot_prod = (normals * normals_gt).sum(-1, keepdim=True)
    dot_prod = dot_prod.clamp(-one_eps, one_eps)
    return torch.arccos(dot_prod) * 180.0 / torch.pi


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1) ** m * 2 ** l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2 ** i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array


def get_sph_harm_mat(deg_view):
    """Generate spherical harmonic coefficient matrix.

    Args:
      deg_view: number of spherical harmonics degrees to use.

    Returns:
      A function for evaluating integrated directional encoding.

    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2 ** (deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    return ml_array, mat


def compute_integrated_dir_enc(ml_array, mat, xyz, kappa_inv):
    """Function returning integrated directional encoding (IDE).

    Args:
      xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
      kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
        Mises-Fisher distribution.

    Returns:
      An array with the resulting IDE.
    """
    x = xyz[..., 0:1]
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]

    # Compute z Vandermonde matrix.
    vmz = torch.cat([z ** i for i in range(mat.shape[0])], axis=-1)

    # Compute x+iy Vandermonde matrix.
    vmxy = torch.cat([(x + 1j * y) ** m for m in ml_array[0, :]], axis=-1)

    # Get spherical harmonics.
    sph_harms = vmxy * torch.matmul(vmz, mat)

    # Apply attenuation function using the von Mises-Fisher distribution
    # concentration parameter, kappa.
    sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
    ide = sph_harms * torch.exp(-sigma * kappa_inv)

    # Split into real and imaginary parts and return
    return torch.cat([torch.real(ide), torch.imag(ide)], axis=-1)


def finitediff_gradient_with_grad(coords, f, eps=0.0005):
    pos_x = coords + torch.tensor([[eps, 0.00, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dx_pos = f(pos_x)
    pos_y = coords + torch.tensor([[0.00, eps, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dy_pos = f(pos_y)
    pos_z = coords + torch.tensor([[0.00, 0.00, eps]], device=coords.device).clamp(-1, 1)
    dist_dz_pos = f(pos_z)

    neg_x = coords + torch.tensor([[-eps, 0.00, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dx_neg = f(neg_x)
    neg_y = coords + torch.tensor([[0.00, -eps, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dy_neg = f(neg_y)
    neg_z = coords + torch.tensor([[0.00, 0.00, -eps]], device=coords.device).clamp(-1, 1)
    dist_dz_neg = f(neg_z)

    gradient = torch.cat([0.5 * (dist_dx_pos - dist_dx_neg) / eps,
                          0.5 * (dist_dy_pos - dist_dy_neg) / eps,
                          0.5 * (dist_dz_pos - dist_dz_neg) / eps],
                         dim=-1)
    return gradient


def asymetric_finitediff_gradient(coords, f, eps=0.0005):
    pos_x = coords + torch.tensor([[eps, 0.00, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dx_pos = f(pos_x)
    pos_y = coords + torch.tensor([[0.00, eps, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dy_pos = f(pos_y)
    pos_z = coords + torch.tensor([[0.00, 0.00, eps]], device=coords.device).clamp(-1, 1)
    dist_dz_pos = f(pos_z)

    neg_x = coords + torch.tensor([[-eps, 0.00, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dx_neg = f(neg_x)
    neg_y = coords + torch.tensor([[0.00, -eps, 0.00]], device=coords.device).clamp(-1, 1)
    dist_dy_neg = f(neg_y)
    neg_z = coords + torch.tensor([[0.00, 0.00, -eps]], device=coords.device).clamp(-1, 1)
    dist_dz_neg = f(neg_z)

    gradient = torch.cat([0.5 * (dist_dx_pos - dist_dx_neg) / eps,
                          0.5 * (dist_dy_pos - dist_dy_neg) / eps,
                          0.5 * (dist_dz_pos - dist_dz_neg) / eps],
                         dim=-1)
    return gradient


def tetrahedron_gradient_with_grad(x, f, eps=0.005):
    """Compute 3D gradient using finite difference (using tetrahedron method).

    Args:
        x (torch.FloatTensor): Coordinate tensor of shape [..., 3]
        f (nn.Module): The function to perform autodiff on.
    """
    h = eps
    k0 = torch.tensor([1.0, -1.0, -1.0], device=x.device, requires_grad=False)
    k1 = torch.tensor([-1.0, -1.0, 1.0], device=x.device, requires_grad=False)
    k2 = torch.tensor([-1.0, 1.0, -1.0], device=x.device, requires_grad=False)
    k3 = torch.tensor([1.0, 1.0, 1.0], device=x.device, requires_grad=False)
    h0 = torch.tensor([h, -h, -h], device=x.device, requires_grad=False)
    h1 = torch.tensor([-h, -h, h], device=x.device, requires_grad=False)
    h2 = torch.tensor([-h, h, -h], device=x.device, requires_grad=False)
    h3 = torch.tensor([h, h, h], device=x.device, requires_grad=False)
    h0 = x + h0
    h1 = x + h1
    h2 = x + h2
    h3 = x + h3
    h0 = k0 * f(h0)
    h1 = k1 * f(h1)
    h2 = k2 * f(h2)
    h3 = k3 * f(h3)
    grad = (h0 + h1 + h2 + h3) / (h * 4.0)
    return grad