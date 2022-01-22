import torch
import torch.nn as nn

def pairwise_distances(x, y, matching_fn='l2', parametric=None):
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances

    elif matching_fn == 'cosine':
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarities = cos(x.unsqueeze(1).expand(n_x, n_y, -1), y.unsqueeze(0).expand(n_x, n_y, -1))
        return 1 - cosine_similarities

    elif matching_fn == 'parametric':
        x_exp = x.unsqueeze(1).expand(n_x, n_y, -1).reshape(n_x*n_y, -1)
        y_exp = y.unsqueeze(0).expand(n_x, n_y, -1).reshape(n_x*n_y, -1)
        
        distances = parametric(torch.cat([x_exp, y_exp], dim=-1))
        return distances.reshape(n_x, n_y)
        
    else:
        raise(ValueError('Unsupported similarity function'))