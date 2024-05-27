
import numpy as np
import torch
import cvxpy as cp


# a simple test


'''

Hammar

'''

# from position_store import chosen_pose

# H_patch = chosen_pose

H_mask = np.array([
        [0,0,0,1],
        [0,0,0,1],
        [0,0,0,1],
        [0,0,0,0]])


def vanilla_inpatint(x_in, H_patch):
    
    x_numpy = x_in.cpu().detach().numpy()

    batch_size = x_numpy.shape[0]
    # repeate  batch_size of H_patch and H_mask

    H_patch_batch = np.repeat(H_patch[np.newaxis, ...], batch_size, axis=0)
    H_mask_batch = np.repeat(H_mask[np.newaxis, ...], batch_size, axis=0)

    x_out = x_numpy * (1- H_mask_batch) +  H_mask_batch * H_patch_batch
#     x_out = H_patch_batch
#     x_out = x_numpy

    # to tensor, float 32
    x_out = torch.tensor(x_out, dtype=torch.float32, device=x_in.device)

    return x_out


def inpaint_opt(x_in, H_patch, threshold = 3e-3):
    x_numpy = x_in.cpu().detach().numpy()
    num_non_zero = np.sum(H_mask)

    batch_size = x_numpy.shape[0]
    # repeate  batch_size of H_patch and H_mask

    H_patch_batch = np.repeat(H_patch[np.newaxis, ...], batch_size, axis=0)
    H_mask_batch = np.repeat(H_mask[np.newaxis, ...], batch_size, axis=0)

    # batch, dim0, dim1
    # flatten dim0 and dim1

    x_flat = x_numpy.reshape(batch_size, -1)
    mask_flat = H_mask_batch.reshape(batch_size, -1)
    patch_flat = H_patch_batch.reshape(batch_size, -1)

    # create optimization variable
    x_hat = cp.Variable(x_flat.shape)
    x_hat.value = x_flat

    # create optimization problem
    # loss 1: inpainting loss
    masked_diff = cp.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = cp.sum_squares(masked_diff) / num_non_zero / batch_size

    # loss 2: in-dist loss
    backward_diff = x_hat - x_flat
    mse_loss_backward = cp.sum_squares(backward_diff)/ batch_size

    # total loss with weight
    total_loss = mse_loss_mask

    # inequality constraint, mse_loss_backward < threshold
    constraint = [mse_loss_backward <= threshold]

    # create optimization problem and solve
    problem = cp.Problem(cp.Minimize(total_loss), constraints=constraint)

    problem.solve()

    # get optimized results
    x_hat = x_hat.value

    # reshape to original shape
    x_hat_ori_dim = x_hat.reshape(x_in.shape)

    # to tensor, float 32
    x_out = torch.tensor(x_hat_ori_dim, dtype=torch.float32, device=x_in.device)

    '''
    masked_diff = np.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = np.mean(masked_diff**2) / num_non_zero / batch_size

    backward_diff = x_hat - x_flat
    mse_loss_backward = np.sum(backward_diff**2)/ batch_size

    print(f'loss patch: {mse_loss_mask}')
    print(f'loss in-dist: {mse_loss_backward}')
    
    '''

    return x_out


