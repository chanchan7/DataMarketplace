import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import str2bool, print_args, colored_text
import sys


def experimental_setting():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    group_dataset.add_argument('--src', type=str, default=None, help='name of the dataset')
    group_dataset.add_argument('--tgt', type=str, default='mnist', help='name of the dataset')
    group_dataset.add_argument('--data-dir', default='../datasets', type=str)
    group_dataset.add_argument('--batch-size', default=100, type=int)
    group_dataset.add_argument('--out-dir', default='../outputs/', type=str, help='Output directory')
    group_dataset.add_argument('--out-feature_dir', default='../outputs/', type=str,
                               help='feature directory')

    group_dataset.add_argument('--n_owners', default=15, type=int, help='number of data owner')
    group_dataset.add_argument('--data_size', default=100, type=int, help='size of dataset each data owner holds')
    group_dataset.add_argument('--public_size', default=7000, help='size of public data')
    group_dataset.add_argument('--valid-size', default=300, type=int, help='size of validation data')
    group_dataset.add_argument('--min_set', default=50, type=int, help='min size of subset: mnist:90, cifar10: 990')
    group_dataset.add_argument('--max_set', default=100, type=int, help='max size of subset: mnist:100, cifar10: 1000')



    # trainer arguments (depends on perturbation)
    group_trainer = init_parser.add_argument_group(f'trainer arguments')
    group_trainer.add_argument('--epochs', default=30, type=int)
    group_trainer.add_argument('--pre_train_epoch', default=40, type=int)
    group_trainer.add_argument('--lr_extractor', default=1e-5, type=float, help='learning rate of extractor')
    group_trainer.add_argument('--lr_deepsets', default=1e-5, type=float, help='learning rate of DeepSets model')
    group_trainer.add_argument('--lr_ext_src', default=1e-5, type=float, help='learning rate of extractor')
    group_trainer.add_argument('--lr_ext_tgt', default=1e-5, type=float, help='learning rate of extractor')
    group_trainer.add_argument('--lr_ext_dis', default=1e-5, type=float, help='learning rate of extractor')


    group_setting = init_parser.add_argument_group(f'experimental setting arguments')
    group_setting.add_argument('--malicious-owners', default=None, type=int, help='Random seed')
    group_setting.add_argument('--adv-owners', default=None, type=int, help='Random seed')
    group_setting.add_argument('--proportion', default=1., help='proportion of adversarial examples in valid data')
    group_setting.add_argument('--epsilon', default=0.3, help='adversarial examples')
    group_setting.add_argument('--unbalance', default=False, help='unbalance data owners')
    group_setting.add_argument('--replicate', default=None, type=int, help='multiple of replicate data')
    group_setting.add_argument('--share', default=0.1, type=float, help='proportion of pre shared data')
    group_setting.add_argument('--prob_hold', default=0.1, help='probability of not fliping data')
    group_setting.add_argument('--noisey', default=False, type=bool, help='add noise to label')

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_trainer.add_argument('--device', help='desired device for training', choices=['cpu', 'cuda'], default='cuda')
    group_expr.add_argument('-s', '--seed', type=int, default=12347, help='initial random seed')
    group_expr.add_argument('--utility-seed', default=123, type=int, help='Random seed when generating deep sets')
    group_expr.add_argument('--tgt-sample-seed', default=456, type=int,
                            help='Random seed when sampling from target dataset')
    group_expr.add_argument('-r', '--repeats', type=int, default=5, help="number of times the experiment is repeated")
    group_expr.add_argument('-o', '--output-dir', type=str, default='./results', help="directory to store the results")
    group_expr.add_argument('--log', type=str2bool, nargs='?', const=True, default=False, help='enable wandb logging')
    group_expr.add_argument('--project-name', type=str, default='Fair_DataMarket', help='wandb project name')

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.device == 'cuda' and not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    return args
