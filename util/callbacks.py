import pytorch_lightning as pl
from hashlib import md5
import os

class LogAndCheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint/logs every N steps
    """

    def __init__(
        self,
        save_step_frequency=50,
        viz_frequency=5,
        log_frequency=5
    ):
        self.save_step_frequency = save_step_frequency
        self.viz_frequency = viz_frequency
        self.log_frequency = log_frequency

    def on_batch_end(self, trainer: pl.Trainer, _):
        global_step = trainer.global_step

        # Saving checkpoint
        if global_step % self.save_step_frequency == 0 and global_step != 0:
            filename = "iter_{}.pth".format(global_step)
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

        # Logging losses
        if global_step % self.log_frequency == 0 and global_step != 0:
            trainer.model.log_current_losses()

        # Image visualization
        if global_step % self.viz_frequency == 0 and global_step != 0:
            trainer.model.log_current_visuals()

class Hash(pl.Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 99:
            print("Hash " + md5(pl_module.state_dict()["netG_B.dec.model.4.conv.weight"].cpu().detach().numpy()).hexdigest())
