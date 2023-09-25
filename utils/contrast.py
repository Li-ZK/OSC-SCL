import torch
import torch.nn.functional as F

def contrast_mse(feature, y, ksai):
    device = feature.device

    feature_matrix = torch.square(feature.unsqueeze(1) - feature.unsqueeze(0)).sum(-1)
    y_matrix = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
    y_matrix[torch.eye(y_matrix.shape[0], dtype=torch.bool)] = -1


    loss_contrast_pos = (torch.where(y_matrix == 1, torch.tensor(1., device=device), torch.tensor(0., device=device)) * feature_matrix.square()).sum()
    loss_contrast_neg = (torch.where(y_matrix == 0, torch.tensor(1., device=device), torch.tensor(0., device=device)) * F.relu(ksai - feature_matrix).square()).sum()
    loss_contrast = (loss_contrast_pos + loss_contrast_neg) / (2 * (feature.shape[0] ** 2 - feature.shape[0]))

    return loss_contrast
