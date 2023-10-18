import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from config import ExpConfig
from data import get_dataloaders

import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append('./ControlNet') # to let ControlNet imports work
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict

cs = ConfigStore.instance()
cs.store(name="base", node=ExpConfig)

@hydra.main(version_base=None, config_path="./configs")
def main(config: ExpConfig):
    # Check config is correctly formatted
    OmegaConf.merge(OmegaConf.structured(ExpConfig), config)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config.model_config_path).cpu()
    embedding = None
    if config.resume_path is not None:
        state_dict = load_state_dict(config.resume_path, location="cpu")
        if 'drop_out_embedding' in state_dict.keys():
            embedding = state_dict['drop_out_embedding']
            del state_dict['drop_out_embedding']
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(load_state_dict(config.model_path, location='cpu'))
    model.learning_rate = config.learning_rate
    model.sd_locked = config.sd_locked
    model.only_mid_control = config.only_mid_control
    model.drop_out_rate = config.drop_out_rate
    if config.one_hot_labels:
        orig_conv = model.control_model.input_hint_block[0]
        model.control_model.input_hint_block[0] = torch.nn.Conv2d(
            22, orig_conv.out_channels, orig_conv.kernel_size, orig_conv.stride, orig_conv.padding, orig_conv.dilation, orig_conv.groups
        )
    if embedding is None:
        model.drop_out_embedding = torch.nn.parameter.Parameter(torch.randn(*config.image_size, 22 if config.one_hot_labels else 3, requires_grad=True))
    else:
        model.drop_out_embedding = embedding
    model.drop_out_text = config.drop_out_text


    # Set up trainer and data loaders
    train_dataloader, val_dataloader = get_dataloaders(config)
    logger = ImageLogger(batch_frequency=config.logger_freq)
    trainer = pl.Trainer(
        precision=32, 
        callbacks=[logger],
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1, 
        default_root_dir=config.log_dir,
        logger=True,
        accelerator='gpu',
        strategy='ddp',
        devices=config.gpus, 
    )

    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    OmegaConf.save(config, f"{trainer.logger.log_dir}/config.yaml")

    # Train!
    trainer.fit(
        model, 
        train_dataloader, 
        val_dataloader, 
        ckpt_path=(None if config.resume_init_only else config.resume_path)
    )

if __name__ == "__main__":
    main()