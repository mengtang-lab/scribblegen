from typing import OrderedDict
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from lib.utils.tools.configer import Configer
from lib.models.model_manager import ModelManager
from lib.loss.loss_manager import LossManager
from segmentor.tools.optim_scheduler import OptimScheduler
from lib.utils.lighting_utils import group_weight, get_parameters, load_state_dict, Evaluator
from data.pascal import VOCSegmentation

class LightningTEL(pl.LightningModule):
    def __init__(self, configer: Configer):
        self.configer = configer

        model_manager = ModelManager(configer)
        self.seg_net = model_manager.semantic_segmentor()

        if self.configer.get('network', 'resume') is not None:
            resume_dict = torch.load(self.configer.get('network', 'resume'))
            if 'state_dict' in resume_dict:
                checkpoint_dict = resume_dict['state_dict']
            elif 'model' in resume_dict:
                checkpoint_dict = resume_dict['model']
            elif isinstance(resume_dict, OrderedDict):
                checkpoint_dict = resume_dict
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(self.configer.get('network', 'resume')))

            if list(checkpoint_dict.keys())[0].startswith('module.'):
                checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}

            # load state_dict
            if hasattr(self.seg_net, 'module'):
                load_state_dict(self.seg_net.module, checkpoint_dict, self.configer.get('network', 'resume_strict'))
            else:
                load_state_dict(self.seg_net, checkpoint_dict, self.configer.get('network', 'resume_strict'))

            if self.configer.get('network', 'resume_continue'):
                self.configer.resume(resume_dict['config_dict'])

        optim_scheduler = OptimScheduler(configer)
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = get_parameters(self.seg_net, configer)
        self.optimizer, self.scheduler = optim_scheduler.init_optimizer(params_group)

        train_ds = VOCSegmentation(configer, split='train')
        val_ds = VOCSegmentation(configer, split='val')
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
        self.train_dataloader = lambda: train_dl
        self.val_dataloader = lambda: val_dl
        self.nclass = 21

        loss_manager = LossManager(configer)
        self.pixel_loss = loss_manager.get_seg_loss()
        self.tree_loss = loss_manager.get_tree_loss()

        self.validation_step_outputs = []  # For metrics
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }
    
    def training_step(self, batch, _):
        aug_imgs = batch["aug_imgs"]
        orig_imgs = batch["orig_imgs"]
        targets = batch["targets"]

        # with torch.cuda.amp.autocast():
        outputs = self.seg_net(aug_imgs)
        # -2: padded pixels;  -1: unlabeled pixels
        unlabeled_RoIs = (targets == -1)
        targets[targets < -1] = -1

        seg_loss = self.pixel_loss([outputs[0]], targets)
        tree_loss = self.tree_loss(outputs[1], orig_imgs, outputs[2], unlabeled_RoIs)
        loss = seg_loss + tree_loss

        self.log("train/seg_loss", seg_loss)
        self.log("train/tree_loss", tree_loss)
        self.log("train/total_loss", loss)
    
    def validation_step(self, batch, _):
        aug_imgs = batch["aug_imgs"]
        orig_imgs = batch["orig_imgs"]
        targets = batch["targets"]

        # with torch.cuda.amp.autocast():
        outputs = self.seg_net(aug_imgs)
        # -2: padded pixels;  -1: unlabeled pixels
        unlabeled_RoIs = (targets == -1)
        targets[targets < -1] = -1

        seg_loss = self.pixel_loss([outputs[0]], targets)
        tree_loss = self.tree_loss(outputs[1], orig_imgs, outputs[2], unlabeled_RoIs)
        loss = seg_loss + tree_loss

        self.log("val/seg_loss", seg_loss)
        self.log("val/tree_loss", tree_loss)
        self.log("val/total_loss", loss)
        
        self.validation_step_outputs.append((outputs[0], targets))

    def on_validation_epoch_end(self):
        evaluator = Evaluator(self.nclass)

        for pred, target in enumerate(self.validation_step_outputs):
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.log('val/mIoU', mIoU)
        self.log('val/Acc', Acc)
        self.log('val/Acc_class', Acc_class)
        self.log('val/fwIoU', FWIoU)
        
        self.validation_step_outputs.clear()
        
