from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """
    This class includes test options with all shared options with train.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--view', type=str, default='ax', help='the view of the pseudo slices as input')
        parser.add_argument('--pad_to_size', type=int, default=-1, help='the size of the slice fed into the network')
        parser.add_argument('--print_stats', action='store_true', help='whether to print the inference metrics or not')
        parser.add_argument('--raw_output', action='store_true', help='whether the inference saves the raw network output')
        # rewrite devalue values
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser
