import torch
import argparse
import dadsh2

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
        mAP = dadsh2.train(
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
    parser.add_argument('--code-length', default='12,16,24', type=str,
                        help='Binary hash code length.(default: 24,32,48)')
    parser.add_argument('--max-iter', default=150, type=int,
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


    # ###################
    # args.max_iter = 5
    # args.num_samples = 200
    args.gpu = 2
    # #############

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    # args.parameter = {'eta': 0.6, 'mu': 1.8, 'gamma': 1.2, 'varphi': 200}
    # args.parameter = {'eta':0.3, 'mu':1.5, 'gamma':1.2, 'varphi':200}    0.7904   0.7969  0.8039
    args.parameter = {'eta':0.6, 'mu':0.6, 'gamma':1.2, 'varphi':200}
    # args.parameter = {'eta':0, 'mu':0, 'gamma':0, 'varphi':200}

# 'eta':0.3, 'mu':0.6, 'gamma':1.2, 'varphi':200     0.7838   0.8092   0.8020
# 'eta':0.6, 'mu':0.6, 'gamma':1.2, 'varphi':200     0.7708   0.8100   0.8055
# 'eta':0.8, 'mu':0.8, 'gamma':1.2, 'varphi':200     0.8121  0.8076  0.7884
# 'eta':0.8, 'mu':0.8, 'gamma':1, 'varphi':200       0.7853  0.7837  0.8048
# 'eta':0.6, 'mu':0.8, 'gamma':1.2, 'varphi':200     0.7863  0.7776
# 'eta':1, 'mu':0.8, 'gamma':1.2, 'varphi':200       0.7584



    return args


if __name__ == '__main__':
    run()
