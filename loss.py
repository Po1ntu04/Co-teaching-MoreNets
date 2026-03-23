import torch
import torch.nn.functional as F
import numpy as np


# Legacy Co-teaching loss kept for compatibility with the original two-model code path.
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    ind_1_sorted = torch.argsort(loss_1).to(y_1.device)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    ind_2_sorted = torch.argsort(loss_2).to(y_2.device)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = max(1, int(remember_rate * len(loss_1_sorted)))

    ind_1_np = ind_1_sorted[:num_remember].detach().cpu().numpy()
    ind_2_np = ind_2_sorted[:num_remember].detach().cpu().numpy()
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_np]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_np]]) / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], reduction="sum") / num_remember
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], reduction="sum") / num_remember

    return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2
