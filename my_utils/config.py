from hpe_library.lib_import import *

# https://dodonam.tistory.com/185 
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_configs(inp=None):
    # Argument parser
    parser = argparse.ArgumentParser(description='Torso Pose Estimator Training')
    parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--gpu', type=str, default="0, 1", help='GPU id')
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='momentum')
    parser.add_argument('--decay_step', type=int, default=100000, help='learning rate decay step')
    parser.add_argument('--out_dir', type=str, default='./checkpoints', help='result directory')
    #parser.add_argument('--log_dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--segment_folder', type=str, default='segments/segments_from_traj_segment_dataset_0.5_[10, 10, 2]_5_2_txtytz_[0.5, 0.5, 0.2]_with_stride1_window5', help='segment folder')
    #parser.add_argument('--segment_file', type=str, default='traj_segment_dataset_0.5_[10, 10, 2]_5_2_txtytz_[0.5, 0.5, 0.2].pickle', help='segment file')
    parser.add_argument('--num_stages', type=int, default=2, help='number of stages in the model')
    parser.add_argument('--skip_connection', type=arg_as_list, default=None, help='use skip connection')
    parser.add_argument('--reprojection', type=bool, default=False, help='use reprojection loss')
    parser.add_argument('--input_list', type=arg_as_list, default=['src_2d_old', 'src_2d_delta'], help='model input')
    parser.add_argument('--output_list', type=arg_as_list, default=['tar_delta_point'], help='model output')
    #parser.add_argument('--', type=, default=, help='')
    #parser.add_argument('--', type=, default=, help='')
    if inp is not None:
        args = parser.parse_args(inp)
    else:
        args = parser.parse_args()

    return args