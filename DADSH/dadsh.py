import torch
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate

from loguru import logger
from models.dadsh_loss import DADSH_Loss
from data.data_loader import sample_dataloader


def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        device,
        lr,
        max_iter,
        max_epoch,
        num_samples,
        batch_size,
        root,
        dataset,
        parameters,
        topk,
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma(float): Hyper-parameters.
        topk(int): Topk k map.

    Returns
        mAP(float): Mean Average Precision.
    """

    # parameters = {'alpha':30, 'eta':2, 'mu':0.2, 'gamma':1,
    #               'omega':1, 'beta':30, 'varphi':200, 'delta':1}
    alpha = parameters['alpha']
    eta = parameters['eta']
    mu = parameters['mu']
    gamma = parameters['gamma']
    omega = parameters['omega']
    beta = parameters['beta']
    varphi = parameters['varphi']
    delta = parameters['delta']

    # Initialization
    model = alexnet.load_model(code_length).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5)
    criterion = DADSH_Loss(code_length, eta, mu, gamma, varphi, delta, device)
    if dataset == 'cifar-10':
        classes = 10
    elif dataset == 'nus-wide-tc21':
        classes = 21

    num_retrieval = len(retrieval_dataloader.dataset)
    U = torch.zeros(num_samples, code_length).to(device)                                # U (m*l, l:code_length)
    B = torch.randn(num_retrieval, code_length).to(device)                              # V (n*l, l:code_length)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)    # Y2 (n*c, c:classes)
    W1 = (2*torch.rand(code_length, classes)-1).to(device)
    W2 = (2*torch.rand(code_length, classes)-1).to(device)

    start = time.time()
    for it in range(max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root, dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(device)    # Y1 (m*c, c:classes)
        S = (train_targets @ retrieval_targets.t() > 0).float()                     # S (m*n, c:classes)
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

        # Training CNN model
        for epoch in range(max_epoch):
            for batch, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()

                F = model(data)             # output (m1*l)
                U[index, :] = F.data       
                cnn_loss = criterion(train_targets[index, :], F.t(), B.t(), S[index, :],
                                     W1, W2, sample_index[index])

                cnn_loss.backward()
                optimizer.step()

        # W1-step
        W1 = update_W(train_targets.t(), U.t(), B.t(), S, W2, 1, alpha, delta, code_length, device)

        # W2-step
        W2 = update_W(retrieval_targets.t(), B.t(), U.t(), S.t(), W1, omega, beta, delta, code_length, device)

        # V-step
        N = ((W1.t() @ U.t()).t() @ W2.t()).t()         # l*m
        expand_U = torch.zeros(B.shape).to(device)      # n*l
        expand_U[sample_index, :] = U
        B = solve_dcc(N, B.t(), expand_U.t(), S, retrieval_targets.t(), W2,
                      code_length, omega, delta, varphi)

        # Total loss
        # iter_loss = calc_loss(train_targets.t(), U.t(), B.t(), S, W1, W2, sample_index,
        #                       code_length, eta, mu, gamma, varphi, delta, device)
        # logger.debug('[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(it+1, max_iter, iter_loss, time.time()-iter_start))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    # Evaluate
    query_code = generate_code(model, query_dataloader, code_length, device)
    mAP = evaluate.mean_average_precision(
        query_code.to(device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_targets,
        device,
        topk,
    )

    # Save checkpoints
    # torch.save(query_code.cpu(), os.path.join('checkpoints', 'query_code.t'))
    # torch.save(B.cpu(), os.path.join('checkpoints', 'database_code.t'))
    # torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join('checkpoints', 'query_targets.t'))
    # torch.save(retrieval_targets.cpu(), os.path.join('checkpoints', 'database_targets.t'))
    # torch.save(model.cpu(), os.path.join('checkpoints', 'model.t'))

    return mAP


def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.reshape(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB


def update_W(Y, U, V, S, W_other, omega, alpha, delta, code_length, device):
    I1 = torch.eye(Y.shape[0], device=device)           # c*c
    I2 = torch.eye(code_length, device=device)          # l*l
    A = omega * U @ U.t() + alpha * I2                  # l*l
    B = delta * U @ U.t()                               # l*l
    C = W_other.t() @ V @ V.t() @ W_other               # c*c
    D = omega*U @ Y.t() + delta*code_length*U @ S @ V.t() @ W_other   # l*c
    vecBD = (torch.pinverse(B) @ D).t().reshape(-1, 1)    # 1*(l*c)
    tmpBA = torch.pinverse(B) @ A
    vecW = torch.pinverse(kronecker(I1, tmpBA) + kronecker(C.t(), I2)) @ vecBD
    return vecW.reshape(-1, code_length).t()


def solve_dcc(N, B, expand_U, S, Y, W, code_length, omega, delta, varphi):
    """
    Solve DCC problem.
    """
    Q = 2 * (omega*W @ Y + delta*code_length*N @ S + varphi * expand_U)   # l*n

    for bit in range(code_length):
        q = Q[bit, :]
        n = N[bit, :]   # N: l*m
        w = W[bit, :]   # W: l*c
        B_prime = torch.cat((B[:bit, :], B[bit+1: ,:]), dim=0)
        N_prime = torch.cat((N[:bit, :], N[bit+1: ,:]), dim=0)
        W_prime = torch.cat((W[:bit, :], W[bit+1: ,:]), dim=0)

        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        tmp = omega*w @ W_prime.t() + delta*n @ N_prime.t() + varphi
        tmp_B = q.t() - 2*(tmp @ B_prime)
        tmp_B = torch.where(tmp_B < -0.33, torch.full_like(tmp_B, -1), tmp_B)
        tmp_B = torch.where(tmp_B > 0.33, torch.full_like(tmp_B, 1), tmp_B)
        tmp_B = torch.where(abs(tmp_B) != 1, torch.full_like(tmp_B, 0), tmp_B)

        B[bit, :] = tmp_B

    return B.t()


def calc_loss(Y, U, V, S, W1, W2, index, code_length, eta, mu, gamma, varphi, delta, device):
    """
    Calculate loss.
    """
    m = index.shape[0]
    discrete_loss1 = eta * torch.norm(U @ torch.eye(m, device=device)) ** 2
    discrete_loss2 = mu * torch.norm((U @ U.t() - 2 * m / 3 * torch.eye(code_length, device=device))) ** 2
    discrete_loss3 = 2 * gamma * (abs(U).mul((1 - U ** 2))).sum()
    quantization_loss1 = torch.norm(Y - W1.t() @ U) ** 2
    quantization_loss2 = varphi * torch.norm(V[:, index] - U) ** 2
    hash_loss = delta * torch.norm(code_length * S - (W1.t() @ U).t() @ (W2.t() @ V)) ** 2

    loss = (discrete_loss1 + discrete_loss2 + discrete_loss3 + hash_loss
            + quantization_loss1 + quantization_loss2) / (V.shape[1] * U.shape[1])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            tmp_code = torch.where(hash_code < -0.33, torch.full_like(hash_code, -1), hash_code)
            tmp_code = torch.where(tmp_code > 0.33, torch.full_like(tmp_code, 1), tmp_code)
            tmp_code = torch.where(abs(tmp_code)!=1, torch.full_like(tmp_code, 0), tmp_code)
            code[index, :] = tmp_code.cpu()

    model.train()
    return code