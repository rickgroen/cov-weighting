from .train_options import TrainOptions, boolstr


class TestOptions(TrainOptions):

    def __init__(self):
        super(TrainOptions).__init__()

    def get_arguments(self, parser):
        parser = TrainOptions.get_arguments(self, parser)

        parser.add_argument('--min_depth',              type=float,     help='minimum depth for evaluation', default=1e-3)
        parser.add_argument('--max_depth',              type=float,     help='maximum depth for evaluation', default=80)
        parser.add_argument('--postprocessing',     type=boolstr,  help='Do post-processing on depth maps', default=True)
        parser.add_argument('--load_final',         type=boolstr,  help='Load final or best trained model', default=False)

        return parser
