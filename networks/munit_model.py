import torch
import itertools
from .base_model import BaseModel
from .backbones import munit as networks
from lab import configs
import munch 

def ModelOptions():
    mo = munch.Munch()
    # Generator
    mo.gen_dim = 64
    mo.style_dim = 8
    mo.gen_activ = 'relu'
    mo.n_downsample = 2
    mo.n_res = 4
    mo.gen_pad_type = 'reflect'
    mo.mlp_dim = 256

    # Discriminiator
    mo.disc_dim = 64
    mo.disc_norm = 'none'
    mo.disc_activ = 'lrelu'
    mo.disc_n_layer = 4
    mo.num_scales = 3 # TODO change for other experiments!
    mo.disc_pad_type = 'reflect'

    # Initialization
    mo.init_type_gen = 'kaiming'
    mo.init_type_disc = 'normal'
    mo.init_gain = 0.02

    # Weights
    mo.lambda_gan = 1
    mo.lambda_rec_image = 10
    mo.lambda_rec_style = 1
    mo.lambda_rec_content = 1
    mo.lambda_rec_cycle = 10
    mo.lambda_vgg = 0.1

    return mo

class MUNITModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'rec_A', 'rec_style_A', 'rec_content_A', 'vgg_A',
                           'D_B', 'G_B', 'cycle_B', 'rec_B', 'rec_style_B', 'rec_content_B', 'vgg_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A_img', 'rec_A_cycle']
        visual_names_B = ['real_B', 'fake_A', 'rec_B_img', 'rec_B_cycle']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']


        self.netG_A = networks.define_G_munit(opt.input_nc, opt.output_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G_munit(opt.input_nc, opt.output_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, self.gpu_ids)

        self.netD_A = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids)

        if opt.lambda_vgg > 0:
            self.instance_norm = torch.nn.InstanceNorm2d(512)
            self.vgg = networks.Vgg16()
            self.vgg.load_state_dict(torch.load('res/vgg_imagenet.pth'))
            self.vgg.to(self.device)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                 weight_decay=0.0001, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        opt_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            weight_decay=0.0001, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        scheduler_G = self.get_scheduler(self.opt, opt_G)
        scheduler_D = self.get_scheduler(self.opt, opt_D)
        return [opt_D, opt_G], [scheduler_D, scheduler_G]

    def reconCriterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A']
        self.real_B = input['B']
        self.image_paths = input['A_paths']

    def __vgg_preprocess(self, batch):
        tensortype = type(batch)
        (r, g, b) = torch.chunk(batch, 3, dim=1)
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        mean = tensortype(batch.data.size()).to(self.device)
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(mean)  # subtract mean
        return batch

    def __compute_vgg_loss(self, img, target):
        img_vgg = self.__vgg_preprocess(img)
        target_vgg = self.__vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instance_norm(img_fea) - self.instance_norm(target_fea)) ** 2)

    def forward(self, style_B_fake = None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Random style sampling
        if style_B_fake is None:
            style_B_fake = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)

        # Encoding
        self.content_A, self.style_A_real = self.netG_A.encode(self.real_A)
        self.fake_B = self.netG_B.decode(self.content_A, style_B_fake)


    def forward_train(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Random style sampling
        self.style_A_fake = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.style_B_fake = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)

        # Encoding
        self.content_A, self.style_A_real = self.netG_A.encode(self.real_A)
        self.content_B, self.style_B_real = self.netG_B.encode(self.real_B)

        # Reconstruction
        self.rec_A_img = self.netG_A.decode(self.content_A, self.style_A_real)
        self.rec_B_img = self.netG_B.decode(self.content_B, self.style_B_real)

        # Cross domain
        self.fake_B = self.netG_B.decode(self.content_A, self.style_B_fake)
        self.fake_A = self.netG_A.decode(self.content_B, self.style_A_fake)

        # Re-encoding everyting
        self.rec_content_B, self.rec_style_A = self.netG_A.encode(self.fake_A)
        self.rec_content_A, self.rec_style_B = self.netG_B.encode(self.fake_B)

        if self.opt.lambda_rec_cycle > 0:
            self.rec_A_cycle = self.netG_A.decode(self.rec_content_A, self.style_A_real)
            self.rec_B_cycle = self.netG_B.decode(self.rec_content_B, self.style_B_real)

    def training_step_D(self):
        with torch.no_grad():
            # Random style sampling
            self.style_A_fake = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
            self.style_B_fake = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)

            # Encoding
            self.content_A, self.style_A_real = self.netG_A.encode(self.real_A)
            self.content_B, self.style_B_real = self.netG_B.encode(self.real_B)

            self.fake_B = self.netG_B.decode(self.content_A, self.style_B_fake)
            self.fake_A = self.netG_A.decode(self.content_B, self.style_A_fake)

        self.loss_D_A = self.netD_A.calc_dis_loss(self.fake_B, self.real_B, self.device) * self.opt.lambda_gan
        self.loss_D_B = self.netD_B.calc_dis_loss(self.fake_A, self.real_A, self.device) * self.opt.lambda_gan

        loss_D = self.loss_D_A + self.loss_D_B
        return loss_D


    def training_step_G(self):
        self.forward_train()
        self.loss_rec_A = self.reconCriterion(self.rec_A_img, self.real_A) * self.opt.lambda_rec_image
        self.loss_rec_B = self.reconCriterion(self.rec_B_img, self.real_B) * self.opt.lambda_rec_image

        self.loss_rec_style_A = self.reconCriterion(self.rec_style_A, self.style_A_fake) * self.opt.lambda_rec_style
        self.loss_rec_style_B = self.reconCriterion(self.rec_style_B, self.style_B_fake) * self.opt.lambda_rec_style

        self.loss_rec_content_A = self.reconCriterion(self.rec_content_A, self.content_A) * self.opt.lambda_rec_content
        self.loss_rec_content_B = self.reconCriterion(self.rec_content_B, self.content_B) * self.opt.lambda_rec_content

        if self.opt.lambda_rec_cycle > 0:
            self.loss_cycle_A = self.reconCriterion(self.rec_A_cycle, self.real_A) * self.opt.lambda_rec_cycle
            self.loss_cycle_B = self.reconCriterion(self.rec_B_cycle, self.real_B) * self.opt.lambda_rec_cycle
        else:
            self.loss_cycle_A = 0
            self.loss_cycle_B = 0

        self.loss_G_A = self.netD_A.calc_gen_loss(self.fake_B, self.device) * self.opt.lambda_gan
        self.loss_G_B = self.netD_B.calc_gen_loss(self.fake_A, self.device) * self.opt.lambda_gan

        if self.opt.lambda_vgg > 0:
            self.loss_vgg_A = self.__compute_vgg_loss(self.fake_A, self.real_B) * self.opt.lambda_vgg
            self.loss_vgg_B = self.__compute_vgg_loss(self.fake_B, self.real_A) * self.opt.lambda_vgg
        else:
            self.loss_vgg_A = 0
            self.loss_vgg_B = 0

        self.loss_G = self.loss_rec_A + self.loss_rec_B + self.loss_rec_style_A + self.loss_rec_style_B + \
            self.loss_rec_content_A + self.loss_rec_content_B + self.loss_cycle_A + self.loss_cycle_B + \
            self.loss_G_A + self.loss_G_B + self.loss_vgg_A + self.loss_vgg_B

        return self.loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):

        self.set_input(batch)
        if optimizer_idx == 0:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.set_requires_grad([self.netG_A, self.netG_B], False)

            return self.training_step_D()
        elif optimizer_idx == 1:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.set_requires_grad([self.netG_A, self.netG_B], True)

            return self.training_step_G()
