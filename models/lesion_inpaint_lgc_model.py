from .loss import *
from utils import *
from .networks import define_net
from .base_model import BaseModel


class LesionInpaintLgcModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new LGC-specific options.
        """
        parser.add_argument('--lambda_lgc', type=float, default=0.1, help='weight for LGC loss')
        parser.add_argument('--lambda_lesion', type=float, default=10, help='weight for lesion area L1 loss')
        parser.add_argument('--lambda_tissue', type=float, default=1, help='weight for valid tissue L1 loss')
        parser.add_argument('--conv_type', type=str, default='gate')
        parser.add_argument('--lgc_layers', type=str, default='enc4_6', help='the layers to compute LGC loss')

        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        """
        Initialize this Lesion Inpaint class.
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['R', 'tissue', 'lesion', 'lgc']
        self.model_names = ['inpaint']
        self.opt = opt

        # define the inpainting network
        self.net_inpaint = define_net(self.opt.input_nc, opt.output_nc, opt.conv_type, opt.norm,
                                              self.opt.init_type, self.opt.init_gain, gpu_ids=self.opt.gpu_ids)

        if self.opt.isTrain:
            # define the loss functions
            self.lesion_loss = L1Loss(weight=opt.lambda_lesion)
            self.tissue_loss = L1Loss(weight=opt.lambda_tissue)
            self.lgc_loss = LGCLoss(lgc_weight=opt.lambda_lgc)

            # define the optimizer
            self.optimizer_inpaint = torch.optim.Adam(self.net_inpaint.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_inpaint)

    def set_input(self, input):
        """
        Read the data of input from dataloader then
        """
        if self.isTrain:
            self.brain = input['brain'].to(self.device)  # get masked brain, i.e. lesion areas have value of 0
            self.lesion = input['lesion'].to(self.device)
            self.gt_img = input['gt'].to(self.device)  # get original brain
            self.image_paths = input['path']  # get image paths
        else:
            self.brain = input['brain'].to(self.device)  # get masked brain, i.e. lesion areas have value of 0
            self.lesion = input['lesion'].to(self.device)


    def forward(self):
        """
        Run forward pass
        """
        # combine pseudo slice and corresponding mask as input
        self.input = torch.cat((self.brain, self.lesion), dim=1)
        self.inpainted, self.inpainted_feat_w_mask = self.net_inpaint(self.input, encoder_only=False, save_feat=True, lgc_layers=self.opt.lgc_layers)

        return self.inpainted

    def backward_inpaint(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # calculate loss given the input and intermediate results

        # calculate reconstruction loss
        self.valid_area = self.gt_img.clone()
        self.valid_area[self.valid_area > 0] = 1
        self.loss_lesion = self.lesion_loss(self.lesion * self.inpainted, self.lesion * self.gt_img)
        self.loss_tissue = self.tissue_loss((1 - self.lesion) * (self.inpainted * self.valid_area)
                                            , (1 - self.lesion) * (self.gt_img * self.valid_area))
        self.loss_L1 = self.loss_lesion + self.loss_tissue


        # calculate LGC loss
        self.inpainted_empty_mask = torch.cat((self.inpainted, torch.zeros(self.inpainted.size(),
                                                                           device=self.inpainted.device)), dim=1)
        self.gt_empty_mask = torch.cat((self.gt_img, torch.zeros(self.gt_img.size(),
                                                                           device=self.gt_img.device)), dim=1)

        self.inpainted_feat_empty_mask = self.net_inpaint(self.inpainted_empty_mask, encoder_only=True)
        self.gt_feat = self.net_inpaint(self.gt_empty_mask, encoder_only=True)
        self.loss_lgc = self.lgc_loss(self.inpainted_feat_w_mask, self.inpainted_feat_empty_mask, self.gt_feat)

        self.loss_R = self.loss_L1 + self.loss_lgc
        self.loss_R.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()

        # update optimizer of the inpainting network
        self.optimizer_inpaint.zero_grad()
        self.backward_inpaint()
        self.optimizer_inpaint.step()


