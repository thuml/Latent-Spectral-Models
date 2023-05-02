import argparse
def get_args():
    parser = argparse.ArgumentParser('Latent Spectral Models', add_help=False)
    # dataset
    parser.add_argument('--data-path', default='./dataset', type=str, help='dataset folder')
    parser.add_argument('--ntotal', default=1200, type=int, help='number of overall data')
    parser.add_argument('--ntrain', default=1000, type=int, help='number of train set')
    parser.add_argument('--ntest', default=200, type=int, help='number of test set')
    parser.add_argument('--in_dim', default=1, type=int, help='input data dimension')
    parser.add_argument('--out_dim', default=1, type=int, help='output data dimension')
    parser.add_argument('--h', default=1, type=int, help='input data height')
    parser.add_argument('--w', default=1, type=int, help='input data width')
    parser.add_argument('--T-in', default=10, type=int,
                        help='input data time points (only for temporal related experiments)')
    parser.add_argument('--T-out', default=10, type=int,
                        help='predict data time points (only for temporal related experiments)')
    parser.add_argument('--h-down', default=1, type=int, help='height downsampe rate of input')
    parser.add_argument('--w-down', default=1, type=int, help='width downsampe rate of input')

    # optimization
    parser.add_argument('--batch-size', default=20, type=int, help='batch size of training')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=501, type=int, help='training epochs')
    parser.add_argument('--step-size', default=100, type=int, help='interval of model save')
    parser.add_argument('--gamma', default=0.5, type=float, help='parameter of learning rate scheduler')

    # Model parameters
    parser.add_argument('--model', default='lsm', type=str, help='model name')
    parser.add_argument('--d-model', default=32, type=int, help='channels of hidden variates')
    parser.add_argument('--num-basis', default=12, type=int, help='number of basis operators')
    parser.add_argument('--num-token', default=4, type=int, help='number of latent tokens')
    parser.add_argument('--patch-size', default='3,3', type=str, help='patch size of different dimensions')
    parser.add_argument('--padding', default='3,3', type=str, help='padding size of different dimensions')

    # save
    parser.add_argument('--model-save-path', default='./checkpoints/', type=str, help='model save path')
    parser.add_argument('--model-save-name', default='lsm.pt', type=str, help='model name')

    return parser.parse_args()