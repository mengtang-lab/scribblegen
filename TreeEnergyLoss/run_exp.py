from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer
from lightning_tel import LightningTEL
from data.pascal import VOCSegmentation

# This file constructs the `configer` required for much of TEL


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--gpus', default=[0], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

    # ***********  Params for TEL.  ***********
    parser.add_argument('--sigma', default=None, type=float,
                        dest='tree_loss:sigma', help='The sigma value for TEL.')

    # ***********  Params for data.  **********
    parser.add_argument('--scribbles', action='store_true', default=False, dest='data:scribbles',
                        help='whether to use scribbles as labels (default: False)')
    parser.add_argument('--base-size', type=int, default=513, dest='data:base_size',
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513, dest='data:crop_size',
                        help='crop image size')
    parser.add_argument('--workers', default=None, type=int,
                        dest='data:workers', help='The number of workers to load data.')
    parser.add_argument('--train_batch_size', default=1, type=int,
                        dest='data:train_batch_size', help='The batch size of training.')
    parser.add_argument('--val_batch_size', default=1, type=int,
                        dest='data:val_batch_size', help='The batch size of validation.')

    parser.add_argument('--aug_scheme', type=str, default='none', dest='data:aug_scheme',
                        choices=['none', 'append', 'sample', 'synth-only', 'replacement'],
                        help='synthetic data augmentation scheme to use (default: none)')
    parser.add_argument('--aug_datasets', type=str, default='normal', dest='data:aug_datasets',
                        help='synthetic datasset to use (default: normal)')
    parser.add_argument('--aug_best_dict', type=str, default=None, dest='data:aug_best_dict',
                        help='optional path to dict of ordering to select samples from (default: None)')
    parser.add_argument('--aug_best_k', type=int, default=1, dest='data:aug_best_k',
                        help='number of the k best images to use, used if aug-best-dict != None')
    parser.add_argument('--replacement_prob', type=float, default=0.0, dest='data:replacement_prob',
                        help='probability of replacing an image, used if aug-scheme=replacement')
    parser.add_argument('--curriculum', type=str, default=None, dest='data:curriculum',
                        help='comma separated list of epochs to switch between datasets (default: None)')
    parser.add_argument('--ssl_split', type=str, default=None, dest='data:ssl_split',
                        help='the SSL split to use (default: None / full dataset)')
    parser.add_argument('--overfit', action='store_true', default=False, dest='data:overfit',
                        help='whether to use exactly 10 images (default: False)')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network:model_name', help='The name of model.')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network:backbone', help='The base network of model.')
    parser.add_argument('--bn_type', default=None, type=str,
                        dest='network:bn_type', help='The BN type of the network.')
    parser.add_argument('--multi_grid', default=None, nargs='+', type=int,
                        dest='network:multi_grid', help='The multi_grid for resnet backbone.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network:pretrained', help='The path to pretrained model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')
    parser.add_argument('--resume_eval_train', type=str2bool, nargs='?', default=True,
                        dest='network:resume_train', help='Whether to validate the training set  during resume.')
    parser.add_argument('--resume_eval_val', type=str2bool, nargs='?', default=True,
                        dest='network:resume_val', help='Whether to validate the val set during resume.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--loss_balance', type=str2bool, nargs='?', default=False,
                        dest='network:loss_balance', help='Whether to balance GPU usage.')

    # ***********  Params for solver.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='optim:optim_method', help='The optim method that used.')
    parser.add_argument('--group_method', default=None, type=str,
                        dest='optim:group_method', help='The group method that used.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='lr:base_lr', help='The learning rate.')
    parser.add_argument('--nbb_mult', default=1.0, type=float,
                        dest='lr:nbb_mult', help='The not backbone mult ratio of learning rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='lr:lr_policy', help='The policy of lr during training.')
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss:loss_type', help='The loss type of the network.')
    parser.add_argument('--is_warm', type=str2bool, nargs='?', default=False,
                        dest='lr:is_warm', help='Whether to warm training.')

    # ***********  Meta params.  **********
    parser.add_argument('--log_dir', default=None, type=str,
                        dest='meta:log_dir', help='The root dir of model save path and tb log path.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='meta:save_iters', help='The saving iters of checkpoint model.')
    parser.add_argument('--save_epoch', default=10, type=int,
                        dest='meta:save_epoch', help='The saving epoch of checkpoint model.')
    parser.add_argument('--epochs', default=-1, type=int,
                        dest='meta:epochs', help='The max epoch of training.')
    parser.add_argument('--iters', default=-1, type=int,
                        dest='solver:max_iters', help='The max iters of training.')
    parser.add_argument('--val_interval', default=2, type=int,
                        dest='meta:val_interval', help='The test interval of validation.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')

    parser.add_argument('REMAIN', nargs='*')

    args_parser = parser.parse_args()

    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)
    assert configer.get('meta', 'log_dir') is not None
    assert (configer.get('meta', 'save_epoch') is None) != (configer.get('meta', 'save_iters') is None), "Exactly one of save_epoch or save_iters must be set"
    assert (configer.get('meta', 'epochs') == -1) != (configer.get('solver', 'max_iters') == -1), "Exactly one of epochs or iters must be set"

    print(configer.get('optim', 'optim_method'))

    train_ds = VOCSegmentation(configer, split='train', overfit=configer.get('data', 'overfit'))
    val_ds = VOCSegmentation(configer, split='val', overfit=configer.get('data', 'overfit'))
    train_dl = DataLoader(
        train_ds, 
        batch_size=configer.get('data', 'train_batch_size'),
        shuffle=True,
        num_workers=configer.get('data', 'workers'),
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=configer.get('data', 'val_batch_size'),
        shuffle=False,
        num_workers=configer.get('data', 'workers'),
    )

    if configer.get('meta', 'epochs') != -1:
        num_iters = len(train_dl) * configer.get('meta', 'epochs') + 1
        configer.update(('solver', 'max_iters'), num_iters)
        print(f"Num epochs: {configer.get('meta', 'epochs')}, Num iters: {configer.get('solver', 'max_iters')}")

    lit_tel_model = LightningTEL(configer)

    checkpointer = ModelCheckpoint(
        filename=f'checkpoint-{{epoch}}', 
        every_n_epochs=configer.get('meta', 'save_epoch'), 
        every_n_train_steps=configer.get('meta', 'save_iters'),
        save_top_k=-1
    )
    latest_saver = ModelCheckpoint(filename='latest-{epoch}', every_n_epochs=1, save_top_k=1)
    trainer = pl.Trainer(
        precision=32, 
        callbacks=[checkpointer, latest_saver],
        max_epochs=configer.get('meta', 'epochs'),
        check_val_every_n_epoch=configer.get('meta', 'val_interval'),
        default_root_dir=configer.get('meta', 'log_dir'),
        logger=True,
        accelerator='gpu',
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        devices=configer.get('gpu'),
    )

    os.makedirs(trainer.logger.log_dir, exist_ok=True)

    trainer.fit(
        lit_tel_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )