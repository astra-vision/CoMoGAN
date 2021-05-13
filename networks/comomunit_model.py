"""
continuous_munit_cyclepoint_residual.py
This is CoMo-MUNIT *logic*, so how the network is trained.
"""

import math
import torch
import itertools
from .base_model import BaseModel
from .backbones import comomunit as networks
import random
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
    mo.lambda_idt = 1
    mo.lambda_Phinet_A = 1
    # Continuous settings
    mo.resblocks_cont = 1
    mo.lambda_physics = 10
    mo.lambda_compare = 10
    mo.lambda_physics_compare = 1

    return mo


class CoMoMUNITModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'rec_A', 'rec_style_B', 'rec_content_A', 'vgg_A', 'phi_net_A',
                           'D_B', 'G_B', 'cycle_B', 'rec_B', 'rec_style_A', 'rec_content_B', 'vgg_B', 'idt_B',
                           'recon_physics', 'phi_net']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['x', 'y', 'rec_A_img', 'rec_A_cycle', 'y_M_tilde', 'y_M']
        visual_names_B = ['y_tilde', 'fake_A', 'rec_B_img', 'rec_B_cycle', 'idt_B_img']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_A', 'D_A', 'G_B', 'D_B', 'DRB', 'Phi_net', 'Phi_net_A']

        self.netG_A = networks.define_G_munit(opt.input_nc, opt.output_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G_munit(opt.output_nc, opt.input_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, self.gpu_ids)

        self.netDRB = networks.define_DRB_munit(opt.resblocks_cont, opt.gen_dim * (2 ** opt.n_downsample), 'instance', opt.gen_activ,
                                                opt.gen_pad_type, opt.init_type_gen, opt.init_gain, self.gpu_ids)
        # define discriminators
        self.netD_A = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids)

        self.netD_B = networks.define_D_munit(opt.input_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids)

        # We use munit style encoder as phinet/phinet_A
        self.netPhi_net = networks.init_net(networks.StyleEncoder(4, opt.input_nc * 2, opt.gen_dim, 2, norm='instance',
                                                                  activ='lrelu', pad_type=opt.gen_pad_type), init_type=opt.init_type_gen,
                                            init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)

        self.netPhi_net_A = networks.init_net(networks.StyleEncoder(4, opt.input_nc, opt.gen_dim, 1, norm='instance',
                                                                    activ='lrelu', pad_type=opt.gen_pad_type), init_type=opt.init_type_gen,
                                              init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)

        # define loss functions
        self.reconCriterion = torch.nn.L1Loss()
        self.criterionPhysics = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

        if opt.lambda_vgg > 0:
            self.instance_norm = torch.nn.InstanceNorm2d(512)
            self.vgg = networks.Vgg16()
            self.vgg.load_state_dict(torch.load('res/vgg_imagenet.pth'))
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),
                                                 self.netDRB.parameters(), self.netPhi_net.parameters(),
                                                 self.netPhi_net_A.parameters()),
                                 weight_decay=0.0001, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        opt_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            weight_decay=0.0001, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        scheduler_G = self.get_scheduler(self.opt, opt_G)
        scheduler_D = self.get_scheduler(self.opt, opt_D)
        return [opt_D, opt_G], [scheduler_D, scheduler_G]

    def set_input(self, input):
        # Input image. everything is mixed so we only have one style
        self.x = input['A']
        # Paths just because maybe they are needed
        self.image_paths = input['A_paths']
        # Desired continuity value which is used to render self.y_M_tilde
        # Desired continuity value which is used to render self.y_M_tilde
        self.phi = input['phi'].float()
        self.cos_phi = input['cos_phi'].float()
        self.sin_phi = input['sin_phi'].float()
        # Term used to train SSN
        self.phi_prime = input['phi_prime'].float()
        self.cos_phi_prime = input['cos_phi_prime'].float()
        self.sin_phi_prime = input['sin_phi_prime'].float()
        # physical model applied to self.x with continuity self.continuity
        self.y_M_tilde = input['A_cont']
        # physical model applied to self.x with continuity self.continuity_compare
        self.y_M_tilde_prime = input['A_cont_compare']
        # Other image, in reality the two will belong to the same domain
        self.y_tilde = input['B']

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


    def forward(self, img, phi = None, style_B_fake = None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Random style sampling
        if style_B_fake is None:
            style_B_fake = torch.randn(img.size(0), self.opt.style_dim, 1, 1).to(self.device)
        if phi is None:
            phi = torch.zeros(1).fill_(random.random()).to(self.device) * math.pi * 2

        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)

        # Encoding
        self.content_A, self.style_A_real = self.netG_A.encode(img)

        features_A = self.netG_B.assign_adain(self.content_A, style_B_fake)
        features_A_real, features_A_physics = self.netDRB(features_A, self.cos_phi, self.sin_phi)
        fake_B = self.netG_B.decode(features_A_real)
        return fake_B

    def training_step_D(self):
        with torch.no_grad():
            # Random style sampling
            self.style_A_fake = torch.randn(self.x.size(0), self.opt.style_dim, 1, 1).to(self.device)
            self.style_B_fake = torch.randn(self.y_tilde.size(0), self.opt.style_dim, 1, 1).to(self.device)

            self.content_A, self.style_A_real = self.netG_A.encode(self.x)
            features_A = self.netG_B.assign_adain(self.content_A, self.style_B_fake)
            features_A_real, features_A_physics = self.netDRB(features_A, self.cos_phi, self.sin_phi)
            self.y = self.netG_B.decode(features_A_real)

            # Encoding
            self.content_B, self.style_B_real = self.netG_B.encode(self.y_tilde)
            features_B = self.netG_A.assign_adain(self.content_B, self.style_A_fake)
            features_B_real, _ = self.netDRB(features_B,
                                             torch.ones(self.cos_phi.size()).to(self.device),
                                             torch.zeros(self.sin_phi.size()).to(self.device)
                                             )
            self.fake_A = self.netG_A.decode(features_B_real)

        self.loss_D_A = self.netD_A.calc_dis_loss(self.y, self.y_tilde) * self.opt.lambda_gan
        self.loss_D_B = self.netD_B.calc_dis_loss(self.fake_A, self.x) * self.opt.lambda_gan

        loss_D = self.loss_D_A + self.loss_D_B
        return loss_D


    def phi_loss_fn(self):
        # the distance between the generated image and the image at the output of the
        # physical model should be zero

        input_zerodistance = torch.cat((self.y, self.y_M_tilde), dim = 1)

        # Distance between generated image and other image of the physical model should be
        # taken from the ground truth value
        input_normaldistance = torch.cat((self.y, self.y_M_tilde_prime), dim = 1)

        # same for this, but this does not depend on a GAN generation so it's used as a regularization term
        input_regolarize = torch.cat((self.y_M_tilde, self.y_M_tilde_prime), dim = 1)
        # essentailly, ground truth distance given by the physical model renderings
        # Cosine distance, we are trying to encode cyclic stuff

        distance_cos = (torch.cos(self.phi) - torch.cos(self.phi_prime)) / 2
        distance_sin = (torch.sin(self.phi) - torch.sin(self.phi_prime)) / 2

        # We evaluate the angle distance and we normalize it in -1/1
        output_zerodistance = torch.tanh(self.netPhi_net(input_zerodistance))#[0])
        output_normaldistance = torch.tanh(self.netPhi_net(input_normaldistance))#[0])
        output_regolarize = torch.tanh(self.netPhi_net(input_regolarize))#[0])

        loss_cos = torch.pow(output_zerodistance[:, 0] - 0, 2).mean()
        loss_cos += torch.pow(output_normaldistance[:, 0] - distance_cos, 2).mean()
        loss_cos += torch.pow(output_regolarize[:, 0] - distance_cos, 2).mean()

        loss_sin = torch.pow(output_zerodistance[:, 1] - 0, 2).mean()
        loss_sin += torch.pow(output_normaldistance[:, 1] - distance_sin, 2).mean()
        loss_sin += torch.pow(output_regolarize[:, 1] - distance_sin, 2).mean()


        # additional terms on the other image generated by the GAN, i.e. something that should resemble exactly
        # the image generated by the physical model
        # This terms follow the same reasoning as before and weighted differently
        input_physics_zerodistance = torch.cat((self.y_M, self.y_M_tilde), dim = 1)
        input_physics_regolarize = torch.cat((self.y_M, self.y_M_tilde_prime), dim = 1)
        output_physics_zerodistance = torch.tanh(self.netPhi_net(input_physics_zerodistance))#[0])
        output_physics_regolarize = torch.tanh(self.netPhi_net(input_physics_regolarize))#[0])

        loss_cos += torch.pow(output_physics_zerodistance[:, 0] - 0, 2).mean() * self.opt.lambda_physics_compare
        loss_cos += torch.pow(output_physics_regolarize[:, 0] - distance_cos,
                          2).mean() * self.opt.lambda_physics_compare
        loss_sin += torch.pow(output_physics_zerodistance[:, 1] - 0, 2).mean() * self.opt.lambda_physics_compare
        loss_sin += torch.pow(output_physics_regolarize[:, 1] - distance_sin,
                          2).mean() * self.opt.lambda_physics_compare

        # Also distance between the two outputs of the gan should be 0
        input_twoheads = torch.cat((self.y_M, self.y), dim = 1)
        output_twoheads = torch.tanh(self.netPhi_net(input_twoheads))#[0])

        loss_cos += torch.pow(output_twoheads[:, 0] - 0, 2).mean()
        loss_sin += torch.pow(output_twoheads[:, 1] - 0, 2).mean()

        loss = loss_cos + loss_sin * 0.5

        return loss

    def training_step_G(self):
        self.style_B_fake = torch.randn(self.y_tilde.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.style_A_fake = torch.randn(self.x.size(0), self.opt.style_dim, 1, 1).to(self.device)

        self.content_A, self.style_A_real = self.netG_A.encode(self.x)
        self.content_B, self.style_B_real = self.netG_B.encode(self.y_tilde)
        self.phi_est = torch.sigmoid(self.netPhi_net_A.forward(self.y_tilde).view(self.y_tilde.size(0), -1)).view(self.y_tilde.size(0)) * 2 * math.pi
        self.estimated_cos_B = torch.cos(self.phi_est)
        self.estimated_sin_B = torch.sin(self.phi_est)

        # Reconstruction
        features_A_reconstruction = self.netG_A.assign_adain(self.content_A, self.style_A_real)
        features_A_reconstruction, _ = self.netDRB(features_A_reconstruction,
                                                   torch.ones(self.estimated_cos_B.size()).to(self.device),
                                                   torch.zeros(self.estimated_sin_B.size()).to(self.device))

        self.rec_A_img = self.netG_A.decode(features_A_reconstruction)

        features_B_reconstruction = self.netG_B.assign_adain(self.content_B, self.style_B_real)
        features_B_reconstruction, _ = self.netDRB(features_B_reconstruction, self.estimated_cos_B, self.estimated_sin_B)

        self.rec_B_img = self.netG_B.decode(features_B_reconstruction)

        # Cross domain
        features_A = self.netG_B.assign_adain(self.content_A, self.style_B_fake)
        features_A_real, features_A_physics = self.netDRB(features_A, self.cos_phi, self.sin_phi)
        self.y_M = self.netG_B.decode(features_A_physics)
        self.y = self.netG_B.decode(features_A_real)

        features_B = self.netG_A.assign_adain(self.content_B, self.style_A_fake)
        features_B_real, _ = self.netDRB(features_B,
                                         torch.ones(self.cos_phi.size()).to(self.device),
                                         torch.zeros(self.sin_phi.size()).to(self.device))
        self.fake_A = self.netG_A.decode(features_B_real)

        self.rec_content_B, self.rec_style_A = self.netG_A.encode(self.fake_A)
        self.rec_content_A, self.rec_style_B = self.netG_B.encode(self.y)

        if self.opt.lambda_rec_cycle > 0:
            features_A_reconstruction_cycle = self.netG_A.assign_adain(self.rec_content_A, self.style_A_real)
            features_A_reconstruction_cycle, _ = self.netDRB(features_A_reconstruction_cycle,
                                                             torch.ones(self.cos_phi.size()).to(self.device),
                                                             torch.zeros(self.sin_phi.size()).to(self.device))
            self.rec_A_cycle = self.netG_A.decode(features_A_reconstruction_cycle)

            features_B_reconstruction_cycle = self.netG_B.assign_adain(self.rec_content_B, self.style_B_real)
            features_B_reconstruction_cycle, _ = self.netDRB(features_B_reconstruction_cycle, self.estimated_cos_B, self.estimated_sin_B)
            self.rec_B_cycle = self.netG_B.decode(features_B_reconstruction_cycle)
        if self.opt.lambda_idt > 0:
            features_B_identity = self.netG_B.assign_adain(self.content_A, torch.randn(self.style_B_fake.size()).to(self.device))
            features_B_identity, _ = self.netDRB(features_B_identity,
                                                 torch.ones(self.estimated_cos_B.size()).to(self.device),
                                                 torch.zeros(self.estimated_sin_B.size()).to(self.device))
            self.idt_B_img = self.netG_B.decode(features_B_identity)


        if self.opt.lambda_idt > 0:
            self.loss_idt_A = 0
            self.loss_idt_B = self.criterionIdt(self.idt_B_img, self.x) * self.opt.lambda_gan * self.opt.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        continuity_angle_fake = torch.sigmoid(self.netPhi_net_A.forward(self.y).view(self.y_tilde.size(0), -1)).view(self.y_tilde.size(0)) * 2 * math.pi

        continuity_cos_fake = 1 - ((torch.cos(continuity_angle_fake) + 1) / 2)
        continuity_cos_gt = 1 - ((torch.cos(self.phi) + 1) / 2)
        continuity_sin_fake = 1 - ((torch.sin(continuity_angle_fake) + 1) / 2)
        continuity_sin_gt = 1 - ((torch.sin(self.phi) + 1) / 2)
        distance_cos_fake = (continuity_cos_fake - continuity_cos_gt)
        distance_sin_fake = (continuity_sin_fake - continuity_sin_gt)

        self.loss_phi_net_A = (distance_cos_fake ** 2) * self.opt.lambda_Phinet_A
        self.loss_phi_net_A += (distance_sin_fake ** 2) * self.opt.lambda_Phinet_A

        self.loss_rec_A = self.reconCriterion(self.rec_A_img, self.x) * self.opt.lambda_rec_image
        self.loss_rec_B = self.reconCriterion(self.rec_B_img, self.y_tilde) * self.opt.lambda_rec_image

        self.loss_rec_style_B = self.reconCriterion(self.rec_style_B, self.style_B_fake) * self.opt.lambda_rec_style
        self.loss_rec_style_A = self.reconCriterion(self.rec_style_A, self.style_A_fake) * self.opt.lambda_rec_style

        self.loss_rec_content_A = self.reconCriterion(self.rec_content_A, self.content_A) * self.opt.lambda_rec_content
        self.loss_rec_content_B = self.reconCriterion(self.rec_content_B, self.content_B) * self.opt.lambda_rec_content

        if self.opt.lambda_rec_cycle > 0:
            self.loss_cycle_A = self.reconCriterion(self.rec_A_cycle, self.x) * self.opt.lambda_rec_cycle
            self.loss_cycle_B = self.reconCriterion(self.rec_B_cycle, self.y_tilde) * self.opt.lambda_rec_cycle
        else:
            self.loss_cycle_A = 0

        self.loss_G_A = self.netD_A.calc_gen_loss(self.y) * self.opt.lambda_gan
        self.loss_G_B = self.netD_B.calc_gen_loss(self.fake_A) * self.opt.lambda_gan

        self.loss_recon_physics = self.opt.lambda_physics * self.criterionPhysics(self.y_M, self.y_M_tilde)
        self.loss_phi_net = self.phi_loss_fn() * self.opt.lambda_compare

        if self.opt.lambda_vgg > 0:
            self.loss_vgg_A = self.__compute_vgg_loss(self.fake_A, self.y_tilde) * self.opt.lambda_vgg
            self.loss_vgg_B = self.__compute_vgg_loss(self.y, self.x) * self.opt.lambda_vgg
        else:
            self.loss_vgg_A = 0
            self.loss_vgg_B = 0

        self.loss_G = self.loss_rec_A + self.loss_rec_style_B + self.loss_rec_content_A + \
                      self.loss_cycle_A + self.loss_G_B + self.loss_vgg_A + \
                      self.loss_rec_B + self.loss_rec_style_A + self.loss_rec_content_B + \
                      self.loss_cycle_B + self.loss_G_A + self.loss_vgg_B + \
                      self.loss_recon_physics + self.loss_phi_net + self.loss_idt_B + self.loss_phi_net_A

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
