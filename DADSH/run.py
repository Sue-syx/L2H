import torch
import argparse
import dadsh

from loguru import logger
from data.data_loader import load_data


def run():
    args = load_config()
    logger.add('logs/{time}.log', rotation='500 MB', level='INFO')
    logger.info(args)

    torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, _, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers,
    )

    for code_length in args.code_length:
        mAP = dadsh.train(
            query_dataloader,
            retrieval_dataloader,
            code_length,
            args.device,
            args.lr,
            args.max_iter,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.parameter,
            args.topk,
        )
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DADSH_PyTorch')
    parser.add_argument('--dataset', default='cifar-10', type=str,
                        help='Dataset name.')
    parser.add_argument('--root', default='../datasets', type=str,
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code-length', default='12,24,32', type=str,
                        help='Binary hash code length.(default: 12,24,32)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--parameter',
                        help='Hyper-parameter.(default: 200)')

    args = parser.parse_args()


    ###################
    # args.max_iter = 5
    # args.num_samples = 200
    args.gpu = 2
    #############

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    # Hyper-parameter
    # args.parameter = {'alpha':50, 'eta':0.8, 'mu':0.2, 'gamma':0.5,
    #                 'omega':1, 'beta':50, 'varphi':30, 'delta':0.5}

    args.parameter = {'alpha':30, 'eta':0.8, 'mu':0.2, 'gamma':0.5,
                    'omega':1, 'beta':30, 'varphi':60, 'delta':0.5}


    return args


if __name__ == '__main__':
    run()
