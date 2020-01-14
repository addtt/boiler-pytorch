import argparse

def parse_args():
    """
    Parse command-line arguments defining experiment settings.

    :return: args: argparse.Namespace with experiment settings
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)

    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        dest='batch_size',
                        help='training batch size')

    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        dest='test_batch_size',
                        help='test batch size')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='learning rate')

    parser.add_argument('--wd',
                        type=float,
                        default=0.0,
                        dest='weight_decay',
                        help='weight decay')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='N',
                        help='random seed')

    parser.add_argument('--tr-log-interv',
                        type=int,
                        default=1000,
                        metavar='N',
                        dest='log_interval',
                        help='number of batches before logging train status')

    parser.add_argument('--ts-log-interv',
                        type=int,
                        default=1000,
                        metavar='N',
                        dest='test_log_interval',
                        help='number of batches before logging test status')

    parser.add_argument('--ll-interv',
                        type=int,
                        default=10000,
                        metavar='N',
                        dest='loglik_interval',
                        help='number of batches before evaluating log likelihood')

    parser.add_argument('--ll-samples',
                        type=int,
                        default=100,
                        metavar='N',
                        dest='loglik_samples',
                        help='number of importance samples to evaluate log likelihood')

    parser.add_argument('--ckpt-interv',
                        type=int,
                        default=100000,
                        metavar='N',
                        dest='checkpoint_interval',
                        help='number of batches before saving model checkpoint')

    parser.add_argument('--nocuda',
                        action='store_true',
                        dest='no_cuda',
                        help='do not use cuda')

    parser.add_argument('--descr',
                        type=str,
                        default='',
                        metavar='STR',
                        dest='additional_descr',
                        help='additional description for experiment name')

    parser.add_argument('--dry-run',
                        action='store_true',
                        dest='dry_run',
                        help='do not save anything to disk')

    parser.add_argument('--resume',
                        type=str,
                        metavar='NAME',
                        default="200114_130927_seed42",
                        help="load the run with this name and resume training")

    args = parser.parse_args()

    assert args.loglik_interval % args.test_log_interval == 0

    return args
