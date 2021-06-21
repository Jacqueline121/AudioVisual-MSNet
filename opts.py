import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='A/V-MSNet options and parameters')

    # Root and results path
    parser.add_argument(
        '--root_path',
        default='./experiments/',
        type=str,
        help='Root directory path of A/V-MSNet experiments')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')

    parser.add_argument(
        '--gpu_devices',
        default='0,1,2,3',
        type=str,
        help='GPU devices ids')
    parser.add_argument(
        '--audiovisual',
        default=True,
        action='store_true',
        help='If True, use the full audiovisual A/V-MSNet, if False use the visual only variant.')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path',
        default='./data/pretrained_models/av-msnet_visual_only/visual_save_60.pth',
        type=str,
        help='Pretrained visual model (.pth)')
    parser.add_argument(
        '--audio_pretrain_path',
        default='./data/pretrained_models/soundnet8.pth',
        type=str,
        help='Pretrained audio model (.pth)')

    # Video frames paths for different datasets
    parser.add_argument(
        '--video_path_tvsum',
        default='./data/video_frames/TVSum',
        type=str,
        help='Directory path of TVSum Videos')
    parser.add_argument(
        '--video_path_movie50',
        default='./data/video_frames/Movie50',
        type=str,
        help='Directory path of Movie50 Videos')

    # Audio paths for different datasets
    parser.add_argument(
        '--audio_path_tvsum',
        default='./data/video_audio/TVSum',
        type=str,
        help='Directory path of TVSum Videos Audio')

    parser.add_argument(
        '--audio_path_movie50',
        default='./data/video_audio/Movie50',
        type=str,
        help='Directory path of Movie50 Videos Audio')

    # Video lists
    parser.add_argument(
        '--annotation_path_tvsum_train',
        default='./data/fold_lists/tvsum_list_train_fps.txt',
        type=str,
        help='Annotation list for TVSum train')
    parser.add_argument(
        '--annotation_path_tvsum_test',
        default='./data/fold_lists/TVSum_list_test_fps.txt',
        type=str,
        help='Annotation list for TVSum test')

    parser.add_argument(
        '--annotation_path_movie50_train',
        default='./data/fold_lists/Movie50_list_train_fps.txt',
        type=str,
        help='Annotation list for Movie50 train')
    parser.add_argument(
        '--annotation_path_movie50_test',
        default='./data/fold_lists/Movie50_list_test_fps.txt',
        type=str,
        help='Annotation list for Movie50 test')


    # Ground truth sumiency maps paths
    parser.add_argument(
        '--gt_path_tvsum',
        default='./data/annotations/TVSum',
        type=str,
        help='gts annotations TVSum')
    parser.add_argument(
        '--gt_path_movie50',
        default='./data/annotations/Movie50',
        type=str,
        help='gts annotations Movie50')
   
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=0.00001, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size',
        default=16, type=int,
        help='Batch size dependent on GPUs memory')
    parser.add_argument(
        '--effective_batch_size',
        default=16, type=int,
        help='Effective batch size independent of GPUs memory.')
    parser.add_argument(
        '--n_epochs',
        default=60,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument(
        '--no_sigmoid_in_test',
        action='store_true',
        help='If true, output for each video is not normalized using sigmoid.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='resnet')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
