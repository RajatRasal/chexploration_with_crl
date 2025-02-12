from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
import torch.nn as nn


class BaseNet(ABC, pl.LightningModule):

    def __init__(
        self,
        num_classes, 
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.inv_loss_coefficient = inv_loss_coefficient
        self.lr_backbone = lr_backbone
        self.lr_disease = lr_disease

        self.fc_disease = None
        self.backbone = None 

    def forward(self, x):
        embedding = self.backbone(x)
        out_disease = self.fc_disease(embedding)
        return embedding, out_disease

    def unpack_batch(self, batch):
        return batch['image'], batch['label'], batch['invariant_attribute']

    def process_batch(self, batch):
        # for non-invariant training
        img, label_class, _ = self.unpack_batch(batch)
        embedding, out_disease = self.forward(img)

        if label_class.ndim==1:
            loss_class = F.cross_entropy(out_disease, label_class)
        else:    
            loss_class = F.binary_cross_entropy_with_logits(out_disease, label_class)
        loss_inv = 0

        return loss_class, loss_inv

    def process_batch_list(self, batch):
        img, label_class, _ = self.unpack_batch(batch)

        loss_class = 0
        loss_inv = 0
        prev_embedding = None
        for i in range(img.shape[0]):  
            embedding, out_disease = self.forward(img[i])  

            if label_class[i].ndim==1:
                loss_class += F.cross_entropy(out_disease, label_class[i])
            else:    
                loss_class += F.binary_cross_entropy_with_logits(out_disease, label_class[i])
            if prev_embedding is not None:
                loss_inv += F.mse_loss(embedding, prev_embedding)
            prev_embedding = embedding

        return loss_class, self.inv_loss_coefficient * loss_inv

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        optim_backbone = torch.optim.Adam(params_backbone, lr=self.lr_backbone)
        optim_disease = torch.optim.Adam(params_disease, lr=self.lr_disease)
        return [optim_disease, optim_backbone]

    def training_step(self, batch, batch_idx):
        invariant_rep = batch['image'].ndim == 5
        batch_processer = self.process_batch_list if invariant_rep else self.process_batch

        loss_class, loss_inv = batch_processer(batch)
        self.log_dict({
            "train_loss_class": loss_class, 
            "train_loss_inv": loss_inv,
        })

        samples = batch['image'] if not invariant_rep else batch['image'][0]
        grid = torchvision.utils.make_grid(samples[0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)

        # Optimise the whole network w.r.t each head separately
        optim_disease, optim_backbone = self.optimizers()

        optim_disease.zero_grad()
        if invariant_rep:
            optim_backbone.zero_grad()
        self.manual_backward(loss_class, retain_graph=True)
        if invariant_rep:
            self.manual_backward(loss_inv)
        optim_disease.step()
        if invariant_rep:
            optim_backbone.step()

    def validation_step(self, batch, batch_idx):
        loss_class, loss_inv = self.process_batch(batch)
        self.log_dict({
            "val_loss_class": loss_class, 
            "val_loss_inv": loss_inv,
        })

    def test_step(self, batch, batch_idx):
        loss_class, loss_inv = self.process_batch(batch)
        self.log_dict({
            "test_loss_class": loss_class, 
            "test_loss_inv": loss_inv,
        })


class ResNet(BaseNet):

    def __init__(
        self,
        num_classes,
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__(
            num_classes,
            inv_loss_coefficient,
            lr_backbone,
            lr_disease,
        )
        self.automatic_optimization = False  # Manual optimization needed
        self.backbone = models.resnet34(weights='IMAGENET1K_V1')
        num_features = self.backbone.fc.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes)
        self.backbone.fc = nn.Identity(num_features)


class DenseNet(BaseNet):

    def __init__(
        self,
        num_classes,
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__(
            num_classes,
            inv_loss_coefficient,
            lr_backbone,
            lr_disease,
        )
        self.automatic_optimization = False  # Manual optimization needed
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes)
        self.backbone.classifier = nn.Identity(num_features)


class ViTB16(BaseNet):

    def __init__(
        self,
        num_classes,
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__(
            num_classes,
            inv_loss_coefficient,
            lr_backbone,
            lr_disease,
        )
        self.automatic_optimization = False  # Manual optimization needed
        self.backbone = models.vit_b_16(weights='IMAGENET1K_V1')
        num_features = self.backbone.heads.head.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes)
        self.backbone.heads.head = nn.Identity(num_features)
