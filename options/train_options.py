from .base_options import BaseOptions, boolstr


class TrainOptions(BaseOptions):

    def __init__(self):
        super(BaseOptions).__init__()

    def get_arguments(self, parser):
        parser = BaseOptions.get_arguments(self, parser)

        parser.add_argument('--epochs',             type=int,   help='number of total epochs to run', default=30)
        parser.add_argument('--learning_rate',      type=float, help='initial learning rate (default: 1e-4)', default=1e-4)
        parser.add_argument('--adjust_lr',          type=boolstr,  help='apply learning rate decay or not', default=True)
        parser.add_argument('--lr_mode',            type=str,   help='Which learning rate scheduler [plateau|polynomial|step]', default='plateau')
        parser.add_argument('--optimizer',          type=str,   help='Optimizer to use [adam|sgd|rmsprop]', default='adam')
        parser.add_argument('--batch_size',         type=int,   help='mini-batch size (default: 8)', default=8)

        parser.add_argument('--img_loss_l1_w',      type=float, help='Weight of the L1 loss component', default=0.25)
        parser.add_argument('--img_loss_ssim_w',    type=float, help='Weight of the SSIM loss component', default=0.25)
        parser.add_argument('--lr_loss_w',          type=float, help='left-right consistency weight', default=0.25)
        parser.add_argument('--disp_grad_loss_w',   type=float, help='disparity smoothness loss weight', default=0.25)

        # Specific to CoV-Weighting
        parser.add_argument('--mean_sort',          type=str,   help='full or decay', default='full')
        parser.add_argument('--mean_decay_param',   type=float, help='What decay to use with mean decay', default=1.0)

        # Specific to GradNorm
        parser.add_argument('--init_gamma',         type=float, help='which alpha to start', default=1.5)

        # Other params
        parser.add_argument('--do_augmentation',    type=boolstr,  help='do augmentation of images or not', default=True)
        parser.add_argument('--augment_parameters', type=str,   help='lowest and highest values for gamma, brightness and color respectively',
                            default=[0.8, 1.2, 0.8, 1.2, 0.8, 1.2])
        parser.add_argument('--norm_layer',         type=str,   help='defines if a normalization layer is used', default='')
        parser.add_argument('--train_ratio',        type=float, help='How much of the training data to use', default=1.0)
        return parser
