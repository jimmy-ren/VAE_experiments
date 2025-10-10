import math
import torch
import numpy as np
import torch.nn.functional as F


def batch_psnr(pred_batch, gt_batch, data_range=1.0, reduction='mean'):
    """
    Compute PSNR for a batch of images and their ground truth.

    Args:
        pred_batch: Batch of predicted images (numpy array or torch tensor)
                    Shape: (batch_size, channels, height, width) or (batch_size, height, width, channels)
        gt_batch: Batch of ground truth images (same format as pred_batch)
        data_range: Maximum possible pixel value (default: 1.0 for normalized images)
        reduction: 'mean' to return average PSNR, 'none' to return individual PSNR values

    Returns:
        PSNR value(s) for the batch
    """

    # Convert to torch tensors if they're numpy arrays
    if isinstance(pred_batch, np.ndarray):
        pred_batch = torch.from_numpy(pred_batch).float()
    if isinstance(gt_batch, np.ndarray):
        gt_batch = torch.from_numpy(gt_batch).float()

    # Ensure both batches are on the same device
    device = pred_batch.device
    gt_batch = gt_batch.to(device)

    # Handle different input formats
    if pred_batch.dim() == 4:
        # If channels last format (batch, height, width, channels), convert to channels first
        if pred_batch.shape[-1] in [1, 3]:
            pred_batch = pred_batch.permute(0, 3, 1, 2)
            gt_batch = gt_batch.permute(0, 3, 1, 2)

    # Calculate MSE for each image in the batch
    mse_per_image = F.mse_loss(pred_batch, gt_batch, reduction='none')
    mse_per_image = mse_per_image.view(pred_batch.shape[0], -1).mean(dim=1)

    # Avoid division by zero
    mse_per_image = torch.clamp(mse_per_image, min=1e-8)

    # Calculate PSNR for each image
    psnr_per_image = 20 * torch.log10(data_range / torch.sqrt(mse_per_image))

    if reduction == 'mean':
        return psnr_per_image.mean().item()
    elif reduction == 'none':
        return psnr_per_image.cpu().numpy()
    else:
        raise ValueError("reduction should be 'mean' or 'none'")

def calculate_psnr(original_img, processed_img, max_pixel_value=255.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Images are expected to be NumPy arrays.
    """
    mse = np.mean((original_img - processed_img) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match, MSE is zero
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

def normalize_mu(mu, assumed_range=(-2, 2)):
    """
    Normalize μ from assumed VAE range to [-1, 1]
    """
    low, high = assumed_range
    # Clamp to prevent outliers from going outside grid
    mu_clamped = torch.clamp(mu, low, high)
    # Linear mapping: [low, high] -> [-1, 1]
    mu_normalized = 2 * (mu_clamped - low) / (high - low) - 1

    return mu_normalized

def hash_and_get_filter(imgs, filter_bank, training=True):
    # compute h and v gradients of LR images for hashing
    h = torch.diff(imgs, dim=3, append=imgs[:, :, :, 0:1])
    v = torch.diff(imgs, dim=2, append=imgs[:, :, 0:1, :])
    # unfold to small patches
    if training:
        unfold = torch.nn.Unfold(kernel_size=(7, 7))
    else:
        unfold = torch.nn.Unfold(kernel_size=(7, 7), padding=(3, 3))
    h_unfolded = unfold(h)
    h_unfolded = torch.permute(h_unfolded, (0, 2, 1))
    h_unfolded = h_unfolded.reshape(-1, 1, 49)
    v_unfolded = unfold(v)
    v_unfolded = torch.permute(v_unfolded, (0, 2, 1))
    v_unfolded = v_unfolded.reshape(-1, 1, 49)
    # compute the structure tensor
    h_v = torch.cat((h_unfolded, v_unfolded), dim=1)
    struct_tensor = torch.matmul(h_v, h_v.permute(0, 2, 1))
    # compute eigenvalues and eigenvectors of the system
    e_val, e_vec = torch.linalg.eigh(struct_tensor.to('cpu'))
    e_val = e_val.to(struct_tensor.device)
    e_vec = e_vec.to(struct_tensor.device)

    # build the 3d coordinates
    # orientation
    orient = torch.atan(e_vec[:, 1, 1] / e_vec[:, 0, 1])
    #orient = torch.atan2(e_vec[:, 1, 1], e_vec[:, 0, 1])

    #grad_orient = torch.atan2(e_vec[:, 1, 1], e_vec[:, 0, 1])  # Range (-π, π]
    #edge_orient = (grad_orient - math.pi / 2) % math.pi  # Subtract 90°, wrap to [0, π)
    #orient = (edge_orient / math.pi) * 2 - 1  # [0, π) -> [-1, 1)

    # normalize from (-π/2, π/2) to (-1, 1)
    orient = orient / (math.pi / 2)
    #orient = grad_orient / math.pi
    #orient = orient % (2*math.pi)  # Remove 180° ambiguity -> [0, π)
    #orient = (orient / math.pi)  - 1  # [0, π) -> [-1, 1)
    #m = max(orient)
    #mm = min(orient)
    # strength
    lambda1 = torch.sqrt(e_val[:, 1])
    lambda2 = torch.sqrt(e_val[:, 0])
    lambda1 = lambda1 + 0.0001
    lambda2 = lambda2 + 0.0001  # avoid nan
    strength = lambda1
    # empirically range of strength [0, 3], normalize to [-1, 1]
    strength = torch.clamp(strength, max=3.0)
    strength = strength / 3 * 2 - 1
    # coherence
    cohe = (lambda1 - lambda2) / (lambda1 + lambda2)
    # empirically range of strength [0, 1], normalize to [-1, 1]
    cohe = cohe * 2 - 1

    orient = orient.view(-1, 1)
    strength = strength.view(-1, 1)
    cohe = cohe.view(-1, 1)
    test = torch.zeros(orient.shape).to(orient.device)
    normalized_coord = torch.cat([strength, cohe, orient], dim=1)
    grid_coords = normalized_coord.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    # duplicate the filter bank for batch_size times
    filter_bank_dup = filter_bank.repeat(orient.size()[0], 1, 1, 1, 1)
    # Perform grid sampling
    sampled_filters = F.grid_sample(
        input=filter_bank_dup,  # [100, 49, 8, 3, 3] - the filter bank
        grid=grid_coords,  # [100, 1, 1, 1, 3] - normalized coordinates
        align_corners=True,  # Important: maps exactly to grid corners
        mode='bilinear',  # Actually trilinear for 3D
        padding_mode='border'  # How to handle out-of-bound coordinates
    )

    final_filters = sampled_filters.squeeze(-1).squeeze(-1).squeeze(-1)
    return final_filters


def combine_filter_batches_pairwise(batch_a, batch_b):
    """
    Combine two batches of filters pairwise: batch_a[i] with batch_b[i]

    Args:
        batch_a: shape (B, H*W)  # e.g., (32, 7*7)
        batch_b: shape (B, H*W)  # e.g., (32, 7*7)

    Returns:
        composite_batch: shape (32, 7*7)
    """
    batch_size = batch_a.shape[0]
    h = int(math.sqrt(batch_a.shape[1]))

    batch_a = batch_a.view(1, batch_size, h, h)
    batch_b = batch_b.view(batch_size, 1, h, h)

    # Approach 1: Using group convolution (efficient)
    composite_batch = F.conv2d(
        batch_a, batch_b,
        padding='same',  # Full convolution
        groups=batch_size,  # Each filter pair convolved separately
    )

    composite_batch = composite_batch.view(batch_size, h*h)
    return composite_batch