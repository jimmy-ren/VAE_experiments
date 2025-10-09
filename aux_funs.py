import math
import torch
import numpy as np
import torch.nn.functional as F

# imgs nchw
# alpha [0 1], time_step >= 0
def noisify(imgs, alpha_schedule, time_step):
    rand_n = torch.randn(imgs.shape)
    # make sure rand_n is stored on the same device as imgs
    rand_n = rand_n.to(imgs.device)

    alpha_bar = torch.prod(alpha_schedule[0:time_step])
    #alpha_bar = alpha_bar[:, None, None, None]
    out = imgs * torch.sqrt(alpha_bar) + rand_n * torch.sqrt(1 - alpha_bar)
    return out, rand_n

# model_outputs n2hw
def model_output_to_onehot(model_outputs):
    ret = torch.zeros(model_outputs.shape)
    condition = model_outputs[:,0,:,:] > model_outputs[:,1,:,:]
    ret[:,0,:,:] = torch.where(condition, 1, 0)
    condition = torch.logical_not(condition)
    ret[:,1,:,:] = torch.where(condition, 1, 0)
    return ret

# prob_imgs n2hw
def prob_to_onehot(prob_imgs):
    prob = prob_imgs[:,0:1,:,:]
    c1 = torch.bernoulli(prob)
    c2 = torch.ones(c1.shape)
    condition = c1 > 0
    c2 = torch.where(condition, 0, c2)
    ret = torch.cat([c1, c2], dim=1)
    return ret

# imgs nchw, one-hot encoding
# Q is a mxm state transition matrix
def noisify_discrete(imgs, Q_bar_set, time_step, return_onehot=True):
    Q = Q_bar_set[time_step, :, :]
    Q = torch.reshape(Q, [Q.shape[0], Q.shape[1], 1, 1])
    imgs = imgs.permute(0, 3, 2, 1) # to nwhc
    Qr = Q.repeat([1, 1, imgs.shape[0], imgs.shape[1]]) # repeat batch_size, w times
    Qr = Qr.permute([2, 3, 0, 1])
    ret = torch.matmul(imgs, Qr)
    # back to nchw
    ret = ret.permute([0, 3, 2, 1])

    if return_onehot:
        # for forward process output should be noisy discrete images (one-hot encoding), not probabilities
        ret = prob_to_onehot(ret)
    return ret

# x_t, x_0, nchw
def reverse_proc_sampling(x_t, x_0, alpha_schedule, time_step, label_type='image'):
    # from time step: t
    # to time step: t-1
    from_ts = time_step + 1
    to_ts = time_step
    alpha_bar_from_ts = torch.prod(alpha_schedule[0:from_ts])
    alpha_bar_to_ts = torch.prod(alpha_schedule[0:to_ts])
    if label_type == 'image':
        mean = (torch.sqrt(alpha_schedule[time_step]) * (1 - alpha_bar_to_ts) * x_t + \
                torch.sqrt(alpha_bar_to_ts) * (1 - alpha_schedule[time_step]) * x_0) / (1 - alpha_bar_from_ts)
    elif label_type == 'noise':
        mean = (1 / torch.sqrt(alpha_schedule[time_step])) * x_t - (1 - alpha_schedule[time_step]) / (torch.sqrt(1 - alpha_bar_from_ts) * (torch.sqrt(alpha_schedule[time_step]))) * x_0
    else:
        # something is wrong
        mean = x_0
    variance = (1 - alpha_schedule[time_step]) * (1 - alpha_bar_to_ts) / (1 - alpha_bar_from_ts)
    rand_n = torch.randn(x_0.shape)
    # make sure rand_n is stored on the same device as x_0
    rand_n = rand_n.to(x_0.device)

    ret = mean + torch.sqrt(variance) * rand_n
    return ret

# x_t, nchw, one-hot encoding
# x_0_prob nchw, in the probability form
# Q_set, Q_bar_set, txmxm state transition matrix
# currently, this function only works for data label with only two classes
def reverse_proc_sampling_discrete(x_t, x_0_prob, Q_set, Q_bar_set, time_step):
    # transpose Q
    Q = Q_set[time_step, :, :]
    Qt = torch.transpose(Q, 0, 1)
    # make the dimension compatible with the function noisify_discrete()
    Qt = torch.reshape(Qt, [1, Q.shape[0], Q.shape[1]])
    p1 = noisify_discrete(x_t, Qt, 0, return_onehot=False)

    s = x_t.shape
    ones_ch = torch.ones([s[0], 1, s[2], s[3]])
    zeros_ch = torch.zeros([s[0], 1, s[2], s[3]])

    x_0_one_hot_class_0 = torch.cat((ones_ch, zeros_ch), dim=1)
    p2_a = noisify_discrete(x_0_one_hot_class_0, Q_bar_set, time_step - 1, return_onehot=False)
    p2_a = p2_a * x_0_prob[:, 0:1, :, :]

    x_0_one_hot_class_1 = torch.cat((zeros_ch, ones_ch), dim=1)
    p2_b = noisify_discrete(x_0_one_hot_class_1, Q_bar_set, time_step - 1, return_onehot=False)
    p2_b = p2_b * x_0_prob[:, 1:2, :, :]

    p2 = p2_a + p2_b

    ret = torch.mul(p1, p2)
    # normalize
    sum = torch.sum(ret, dim=1, keepdim=True)
    ret = ret / sum
    ret = prob_to_onehot(ret)
    return ret

class BinaryOneHotTransform():
    def __call__(self, sample):
        i = torch.ones([1, sample.shape[1], sample.shape[2]])
        j = torch.zeros([1, sample.shape[1], sample.shape[2]])
        condition = sample == 0
        # Set elements satisfying the condition to a new value (e.g., 0)
        i = torch.where(condition, 0, i)
        j = torch.where(condition, 1, j)
        ret = torch.cat([i, j], dim=0)
        return ret

def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    """
    Transform a scalar time-step into a vector representation of size t_emb_dim.

    :param time_steps: 1D tensor of size -> (Batch,)
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)

    :return tensor of size -> (B, t_emb_dim)
    """
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."

    factor = 2 * torch.arange(start=0,
                              end=t_emb_dim // 2,
                              dtype=torch.float32,
                              device=time_steps.device
                              ) / (t_emb_dim)

    factor = 10000 ** factor

    t_emb = time_steps[:, None]  # B -> (B, 1)
    t_emb = t_emb / factor  # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)  # (B , t_emb_dim)

    return t_emb

# matrices, nwh
def cumulative_matrix_mul(matrices):
    num_of_matrices = matrices.shape[0]
    cumulative_matrices = torch.zeros(matrices.shape)
    current_product = matrices[0, :, :]
    for i in range(num_of_matrices):
        cumulative_matrices[i, :, :] = current_product
        if i != num_of_matrices - 1:
            current_product = torch.matmul(current_product, matrices[i+1, :, :])

    return cumulative_matrices


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

