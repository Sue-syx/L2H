import torch
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate

from loguru import logger
from models.dadsh2_loss import DADSH2_Loss
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

    # parameters = {'eta':2, 'mu':0.2, 'gamma':1, 'varphi':200}
    eta = parameters['eta']
    mu = parameters['mu']
    gamma = parameters['gamma']
    varphi = parameters['varphi']

    # Initialization
    model = alexnet.load_model(code_length).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5)
    criterion = DADSH2_Loss(code_length, eta, mu, gamma, varphi, device)

    num_retrieval = len(retrieval_dataloader.dataset)
    U = torch.zeros(num_samples, code_length).to(device)                                # U (m*l, l:code_length)
    B = torch.randn(num_retrieval, code_length).to(device)                              # V (n*l, l:code_length)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)    # Y2 (n*c, c:classes)


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
                cnn_loss = criterion(F, B, S[index, :], sample_index[index])

                cnn_loss.backward()
                optimizer.step()

        # Update B
        expand_U = torch.zeros(B.shape).to(device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, varphi)

        # Total loss
        # iter_loss = calc_loss(U, B, S, code_length, sample_index, gamma)

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
    torch.save(query_code.cpu(), os.path.join('checkpoints', 'query_code.t'))
    torch.save(B.cpu(), os.path.join('checkpoints', 'database_code.t'))
    torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join('checkpoints', 'query_targets.t'))
    torch.save(retrieval_targets.cpu(), os.path.join('checkpoints', 'database_targets.t'))
    torch.save(model.cpu(), os.path.join('checkpoints', 'model.t'))

    return mAP


def solve_dcc(B, U, expand_U, S, code_length, varphi):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + varphi * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

        tmp_B = q.t() - B_prime @ U_prime.t() @ u.t()
        tmp_B = torch.where(tmp_B < -0.33, torch.full_like(tmp_B, -1), tmp_B)
        tmp_B = torch.where(tmp_B > 0.33, torch.full_like(tmp_B, 1), tmp_B)
        tmp_B = torch.where(abs(tmp_B) != 1, torch.full_like(tmp_B, 0), tmp_B)

        B[:, bit] = tmp_B

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

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