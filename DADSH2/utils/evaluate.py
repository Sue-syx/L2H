import torch
import time


def mean_average_precision(query_code_m,
                           database_code_m,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    # query_code: m*l
    # database_code: n*l
    code_length = query_code_m.shape[1]
    m = query_code_m.shape[0]
    n = database_code_m.shape[0]

    query_code = torch.zeros(m, 2*code_length)
    for i, data_i in enumerate(query_code_m.chunk(code_length, 1)):
        tmp1 = query_code_m[:, i]
        tmp_1 = torch.where(tmp1 == 0, torch.full_like(tmp1, 1), tmp1)
        tmp2 = -query_code_m[:, i]
        tmp_2 = torch.where(tmp2 == 0, torch.full_like(tmp2, 1), tmp2)
        query_code[:, 2 * i] = tmp_1
        query_code[:, 2 * i + 1] = tmp_2

    database_code = torch.zeros(n, 2*code_length)
    for i, data_i in enumerate(database_code_m.chunk(code_length, 1)):
        tmp1 = database_code_m[:, i]
        tmp_1 = torch.where(tmp1 == 0, torch.full_like(tmp1, 1), tmp1)
        tmp2 = -database_code_m[:, i]
        tmp_2 = torch.where(tmp2 == 0, torch.full_like(tmp2, 1), tmp2)
        database_code[:, 2 * i] = tmp_1
        database_code[:, 2 * i + 1] = tmp_2

    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP
