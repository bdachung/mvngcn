import torch

def rowSampling(weight, size):
    # print(weight.shape)
    sorted_weight, indices = torch.sort(weight)
    return sorted_weight[-size:]

def sum_log_weight(weight):
    weight = torch.log(weight)
    return torch.sum(weight)

def loss_sparsity(weight, lambda2, lambda3):
    m = weight.size(0)
    v_ones = torch.ones((1,m))
    loss1 = torch.matmul(torch.matmul(v_ones,torch.log(torch.square(weight))), torch.t(v_ones))
    # print(loss1)
    loss2 = torch.sum(torch.square(weight))
    # print(loss2)
    return (-loss1*lambda2/m + loss2*lambda3/m)[0][0]
