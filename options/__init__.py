"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""

from argparse import ArgumentParser as AP
from .train_options import TrainOptions
from .log_options import LogOptions
from networks import get_model_options
from data import get_dataset_options
import munch


def get_options(cmdline_opt):

    bo = munch.Munch()
    # Set the number of channels of input image
    # Set the number of channels of output image
    bo.input_nc = 3
    bo.output_nc = 3
    bo.gpu_ids = cmdline_opt.gpus
    # Dataset options
    bo.dataroot = cmdline_opt.path_data
    bo.dataset_mode = cmdline_opt.data_importer
    bo.model = cmdline_opt.model
    # Scheduling policies
    bo.lr = cmdline_opt.learning_rate
    bo.lr_policy = cmdline_opt.scheduler_policy
    bo.decay_iters_step = cmdline_opt.decay_iters_step
    bo.decay_step_gamma = cmdline_opt.decay_step_gamma

    opts = []
    opts.append(get_model_options(bo.model)())
    opts.append(get_dataset_options(bo.dataset_mode)())
    opts.append(LogOptions())
    opts.append(TrainOptions())

    # Checks for Nones
    opts = [x for x in opts if x]
    for x in opts:
        bo.update(x)
    return bo
