"""
base_model.py
Abstract definition of a model, where helper functions as image extraction and gradient propagation are defined.
"""

from collections import OrderedDict
from abc import abstractmethod

import pytorch_lightning as pl
from torch.optim import lr_scheduler

from torchvision.transforms import ToPILImage

class BaseModel(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.save_hyperparameters()
        self.schedulers = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input):
        pass

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def compute_visuals(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        return lr

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = (getattr(self, name).detach() + 1) / 2
        return visual_ret

    def log_current_losses(self):
        losses = '\n'
        for name in self.loss_names:
            if isinstance(name, str):
                loss_value = float(getattr(self, 'loss_' + name))
                self.logger.log_metrics({'loss_{}'.format(name): loss_value}, self.trainer.global_step)
                losses += 'loss_{}={:.4f}\t'.format(name, loss_value)
        print(losses)

    def log_current_visuals(self):
        visuals = self.get_current_visuals()
        for key, viz in visuals.items():
            self.logger.experiment.add_image('img_{}'.format(key), viz[0].cpu(), self.trainer.global_step)

    def get_scheduler(self, opt, optimizer):
        if opt.lr_policy == 'linear':
            def lambda_rule(iter):
                lr_l = 1.0 - max(0, self.trainer.global_step - opt.static_iters) / float(opt.decay_iters + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.decay_iters_step, gamma=0.5)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def print_networks(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))

    def get_optimizer_dict(self):
        return_dict = {}
        for index, opt in enumerate(self.optimizers):
            return_dict['Optimizer_{}'.format(index)] = opt
        return return_dict

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
