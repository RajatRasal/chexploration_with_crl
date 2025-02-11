import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
from dotenv import load_dotenv

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

import json
from abc import ABC

load_dotenv()

image_size = (224, 224)
num_classes_disease = 2
batch_size = 32
epochs = 10
num_workers = 4
chexpert_img_data_dir = os.getenv("CHEXPERT_FOLDER")
mimic_img_data_dir = os.getenv("MIMIC_FOLDER")


class CheXpertDataset(Dataset):

    def __init__(
        self,
        csv_file_img,
        image_size, 
        img_data_dir,
        augmentation=False, 
        pseudo_rgb=True,
        use_cache=False,
        nsamples=2,
        invariant_sampling=False,
        protected_race_set=[0, 1],
    ):

        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.invariant_sampling = invariant_sampling
        self.use_cache = use_cache
        self.protected_race_set = protected_race_set

        self.labels = ['No Finding', 'Pleural Effusion']

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        if self.invariant_sampling:
            # We have 3 races: 0, 1, 2
            protected_race_attributes = np.unique(self.data['race_label'].values)
            protected_race_counts = np.ones_like(protected_race_attributes)
            self.attribute_wise_samples = {}

            for race in np.unique(protected_race_attributes):
                self.attribute_wise_samples[race] = {}
                protected_race_counts[race] = np.sum(1. * (self.data['race_label'].values == race))

            protected_race_probs = 1. / protected_race_counts

            self.nsamples = nsamples
            self.protected_race_probs = protected_race_probs / np.sum(protected_race_probs)

            self.label_list = []

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            # Convert pandas rows into a dictionary which can be retrieved in __getitem__
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label_disease = [
                float(self.data.loc[idx, label] == 1)
                for label in self.labels
            ]
            img_label_race = int(self.data.loc[idx, 'race_label'])
            # Only include races in the protected race set
            if img_label_race not in self.protected_race_set:
                continue

            sample = {
                'image_path': img_path,
                'label_disease': img_label_disease,
                'protected_attribute': img_label_race,
            }
            self.samples.append(sample)

            if self.invariant_sampling:
                # Build race invariant sets for the same diseases
                disease_key = ''.join(map(str, img_label_disease))
                if disease_key not in self.attribute_wise_samples[img_label_race]:
                    self.attribute_wise_samples[img_label_race][disease_key] = []

                self.attribute_wise_samples[img_label_race][disease_key].append(sample)

                if disease_key not in self.label_list:
                    self.label_list.append(disease_key)

        if self.use_cache:
            self.cache = {}

    def __len__(self):
        return len(self.samples)

    def getitem_inv(self, item):
        samples = self.get_sample(item)

        images = []
        labels_disease = []
        labels_race = []

        for sample in samples:
            image = sample['image'].unsqueeze(0)
            label_disease = torch.tensor(sample['label_disease'])
            label_race = torch.tensor(sample['protected_attribute'])

            if self.do_augment:
                image = self.augment(image)

            if self.pseudo_rgb:
                image = image.repeat(3, 1, 1)

            images.append(image.unsqueeze(0))
            labels_disease.append(label_disease.unsqueeze(0))
            labels_race.append(label_race.unsqueeze(0))

        images = torch.cat(images, 0)
        labels_disease = torch.cat(labels_disease, 0)
        labels_race = torch.cat(labels_race, 0)

        return {
            'image': images,
            'label_disease': labels_disease, 
            'protected_attribute': labels_race,
        }

    def getitem(self, item):
        sample = self.get_sample(item)
        image = sample['image'].unsqueeze(0)
        label_disease = torch.tensor(sample['label_disease'])
        label_race = torch.tensor(sample['protected_attribute'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {
            'image': image, 
            'label_disease': label_disease, 
            'protected_attribute': label_race,
        }

    def __getitem__(self, item):
        return self.getitem_inv(item) if self.invariant_sampling else self.getitem(item)
        
    def get_sample(self, item):
        if self.invariant_sampling:
            return self.get_samples(item)
        else:
            sample = self.samples[item]
            image = None

            if self.use_cache:
                if sample['image_path'] in self.cache:
                    image = self.cache[sample['image_path']]
            
            if image is None:
                image = self._read_image(sample['image_path'])
                if self.use_cache:
                    self.cache[sample['image_path']] = image
                
            return {
                'image': image, 
                'label_disease': sample['label_disease'], 
                'protected_attribute': sample['protected_attribute'],
            }

    def get_samples(self, item):
        np.random.seed(item)

        # Sample a disease
        disease = np.random.choice(self.label_list)
        prob = self.protected_race_probs[self.protected_race_set]
        prob = prob / np.sum(prob)  # renormalising
        race = np.random.choice(
            self.protected_race_set,
            self.nsamples,
            p=prob,
        )
        
        # Get samples for the chosen disease from the race invariant set.
        # This allows ensures that representations for the same disease
        # are invariant to race when trained in a fashion akin to https://arxiv.org/abs/2106.04619.
        info = []
        for si in range(self.nsamples):
            sample = np.random.choice(self.attribute_wise_samples[race[si]][disease])

            image = None
            if self.use_cache:
                if sample['image_path'] in self.cache:
                    image = self.cache[sample['image_path']]
            
            if image is None:
                image = self._read_image(sample['image_path'])
                if self.use_cache:
                    self.cache[sample['image_path']] = image
                
            info.append({
                'image': image, 
                'label_disease': sample['label_disease'], 
                'protected_attribute': sample['protected_attribute'],
            })  
        return info

    def _read_image(self, image_path):
        return torch.from_numpy(imread(image_path).astype(np.float32))


class CheXpertDataModule(pl.LightningDataModule):

    def __init__(
        self,
        csv_train_img, 
        csv_val_img, 
        csv_test_img, 
        image_size, 
        pseudo_rgb, 
        batch_size, 
        num_workers, 
        img_data_dir, 
        use_cache=False,
        nsamples=2,
        invariant_sampling=False,
        protected_race_set_train=[0, 1, 2, 3],
        protected_race_set_test=[0, 1, 2, 3],
    ):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nsamples = nsamples
        self.use_cache = use_cache
        self.invariant_sampling = invariant_sampling
        self.protected_race_set_train = protected_race_set_train
        self.protected_race_set_test = protected_race_set_test

        self.train_set = CheXpertDataset(
            self.csv_train_img, 
            self.image_size, 
            img_data_dir,
            augmentation=True, 
            pseudo_rgb=pseudo_rgb,
            use_cache=self.use_cache,
            nsamples=self.nsamples,
            invariant_sampling=self.invariant_sampling,
            protected_race_set=self.protected_race_set_train,
        )
        self.val_set = CheXpertDataset(
            self.csv_val_img, 
            self.image_size, 
            img_data_dir,
            augmentation=False, 
            pseudo_rgb=pseudo_rgb,
            nsamples=1,
            invariant_sampling=False,
            protected_race_set=self.protected_race_set_train,
        )
        self.test_set = CheXpertDataset(
            self.csv_test_img,
            self.image_size,
            img_data_dir,
            augmentation=False,
            pseudo_rgb=pseudo_rgb,
            nsamples=1,
            invariant_sampling=False,
            protected_race_set=self.protected_race_set_test,
        )

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)



class BaseNet(ABC, pl.LightningModule):

    def __init__(
        self,
        num_classes_disease, 
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__()
        self.num_classes_disease = num_classes_disease
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
        return batch['image'], batch['label_disease'], batch['protected_attribute']

    def process_batch(self, batch):
        # for non-invariant training
        img, lab_disease, _ = self.unpack_batch(batch)
        embedding, out_disease = self.forward(img)

        loss_disease = F.binary_cross_entropy_with_logits(out_disease, lab_disease)
        loss_inv = 0

        return loss_disease, loss_inv

    def process_batch_list(self, batch):
        img, lab_disease, _ = self.unpack_batch(batch)

        loss_disease = 0
        loss_inv = 0
        prev_embedding = None
        for i in range(img.shape[0]):  
            embedding, out_disease = self.forward(img[i])  
            loss_disease += F.binary_cross_entropy_with_logits(out_disease, lab_disease[i])
            if prev_embedding is not None:
                loss_inv += F.mse_loss(embedding, prev_embedding)
            prev_embedding = embedding

        return loss_disease, self.inv_loss_coefficient * loss_inv

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        optim_backbone = torch.optim.Adam(params_backbone, lr=self.lr_backbone)
        optim_disease = torch.optim.Adam(params_disease, lr=self.lr_disease)
        return [optim_disease, optim_backbone]

    def training_step(self, batch, batch_idx):
        invariant_rep = len(batch['image'].shape) == 5
        batch_processer = self.process_batch_list if invariant_rep else self.process_batch

        loss_disease, loss_inv = batch_processer(batch)
        self.log_dict({
            "train_loss_disease": loss_disease, 
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
        self.manual_backward(loss_disease, retain_graph=True)
        if invariant_rep:
            self.manual_backward(loss_inv)
        optim_disease.step()
        if invariant_rep:
            optim_backbone.step()

    def validation_step(self, batch, batch_idx):
        loss_disease, loss_inv = self.process_batch(batch)
        self.log_dict({
            "val_loss_disease": loss_disease, 
            "val_loss_inv": loss_inv,
        })

    def test_step(self, batch, batch_idx):
        loss_disease, loss_inv = self.process_batch(batch)
        self.log_dict({
            "test_loss_disease": loss_disease, 
            "test_loss_inv": loss_inv,
        })


class ResNet(BaseNet):

    def __init__(
        self,
        num_classes_disease,
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__(
            num_classes_disease,
            inv_loss_coefficient,
            lr_backbone,
            lr_disease,
        )
        self.automatic_optimization = False  # Manual optimization needed
        self.backbone = models.resnet34(weights='IMAGENET1K_V1')
        num_features = self.backbone.fc.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.fc = self.fc_connect


class DenseNet(BaseNet):

    def __init__(
        self,
        num_classes_disease,
        inv_loss_coefficient,
        lr_backbone=0.001,
        lr_disease=0.001,
    ):
        super().__init__(
            num_classes_disease,
            inv_loss_coefficient,
            lr_backbone,
            lr_disease,
        )
        self.automatic_optimization = False  # Manual optimization needed
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect


def test(model, data_loader, device):
    model.eval()
    logits_disease = []
    preds_disease = []
    embeddings = []
    targets_disease = []
    targets_protected_attributes = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, protected_attribute = batch['image'].to(device), batch['label_disease'].to(device), batch['protected_attribute'].to(device)
            embedding, out_disease = model(img)

            pred_disease = torch.sigmoid(out_disease)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            embeddings.append(embedding)
            targets_protected_attributes.append(protected_attribute)

        embeddings = torch.cat(embeddings, dim=0)
        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_protected_attributes = torch.cat(targets_protected_attributes, dim=0)

        info = {'global': {}, 'sub-group': {}}

        for i in range(0, num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t).item()
            info['global'][i] = int(c)

        for j in torch.unique(targets_protected_attributes):
            info['sub-group'][int(j.item())] = {}
            for i in range(0, num_classes_disease):
                t = targets_disease[targets_protected_attributes == j] == i
                c = torch.sum(t).item()
                info['sub-group'][int(j.item())][i] = int(c)

        print(json.dumps(info, indent=4))

    return (
        info,
        embeddings.cpu().numpy(),
        preds_disease.cpu().numpy(),
        targets_disease.cpu().numpy(),
        logits_disease.cpu().numpy(),
        targets_protected_attributes.cpu().numpy(),
    )


def main(hparams):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)

    # model
    if hparams.model_type == 'densenet':
        model_type = DenseNet
    elif hparams.model_type == 'resnet':
        model_type = ResNet
    else:
        raise NotImplementedError

    model = model_type(
        num_classes_disease=num_classes_disease,
        inv_loss_coefficient=hparams.inv_loss_coefficient,
    )

    # Create output directory
    logdir = "logs_test" if hparams.test else "logs"
    if hparams.dataset_train == hparams.dataset_test:
        # attribute_transfer
        out_dir = f'{logdir}/{model_type.__name__}/{hparams.dataset_train}/{hparams.protected_race_set_train}_{hparams.protected_race_set_test}'

        if hparams.dataset_train == 'chexpert':
            csv_train_img='datafiles/chexpert/chexpert.sample.train.csv'
            csv_val_img='datafiles/chexpert/chexpert.sample.val.csv'
            csv_test_img='datafiles/chexpert/chexpert.sample.test.csv'
            img_data_dir=chexpert_img_data_dir
        else:
            csv_train_img='datafiles/mimic/mimic.sample.train.csv'
            csv_val_img='datafiles/mimic/mimic.sample.val.csv'
            csv_test_img='datafiles/mimic/mimic.sample.test.csv'
            img_data_dir=mimic_img_data_dir
        
        data_train = CheXpertDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            img_data_dir=img_data_dir,
            image_size=image_size,
            pseudo_rgb=True,
            batch_size=batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_train,
        )
        data_test = CheXpertDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            img_data_dir=img_data_dir,
            image_size=image_size,
            pseudo_rgb=True,
            batch_size=batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test, # Different from data_train module
        )
    else:
        # dataset transfer
        out_dir = f'{logdir}/{model_type.__name__}/{hparams.dataset_train}_{hparams.dataset_test}'

        # train set
        if hparams.dataset_train == 'chexpert':
            csv_train_img='datafiles/chexpert/chexpert.sample.train.csv'
            csv_val_img='datafiles/chexpert/chexpert.sample.val.csv'
            csv_test_img='datafiles/chexpert/chexpert.sample.test.csv'
            img_data_dir=chexpert_img_data_dir
        else:
            csv_train_img='datafiles/mimic/mimic.sample.train.csv'
            csv_val_img='datafiles/mimic/mimic.sample.val.csv'
            csv_test_img='datafiles/mimic/mimic.sample.test.csv'
            img_data_dir=mimic_img_data_dir
        
        data_train = CheXpertDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            img_data_dir=img_data_dir,
            image_size=image_size,
            pseudo_rgb=True,
            batch_size=batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test,
        )

        # Test set 
        if hparams.dataset_test == 'chexpert':
            csv_train_img='datafiles/chexpert/chexpert.sample.train.csv'
            csv_val_img='datafiles/chexpert/chexpert.sample.val.csv'
            csv_test_img='datafiles/chexpert/chexpert.sample.test.csv'
            img_data_dir=chexpert_img_data_dir
        else:
            csv_train_img='datafiles/mimic/mimic.sample.train.csv'
            csv_val_img='datafiles/mimic/mimic.sample.val.csv'
            csv_test_img='datafiles/mimic/mimic.sample.test.csv'
            img_data_dir=mimic_img_data_dir

        data_test = CheXpertDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            image_size=image_size,
            img_data_dir=img_data_dir,
            pseudo_rgb=True,
            batch_size=batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test,
        )

    if hparams.invariant_sampling:
        out_dir = os.path.join(out_dir, f"invariant_nsamples_{hparams.nsamples}")
    else:
        out_dir = os.path.join(out_dir, "non_invariant")
    os.makedirs(out_dir, exist_ok=True)

    temp_dir = os.path.join(out_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    for idx in range(0, 5):
        sample = data_train.val_set.get_sample(idx)
        imsave(
            os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'),
            sample['image'].numpy().astype(np.uint8),
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_disease",
        mode="min",
        save_last=True,
    )

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        max_epochs=epochs,
        devices=hparams.gpus,
        logger=TensorBoardLogger(out_dir),
        max_steps=5 if hparams.test else -1,
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data_train)

    if hparams.test:
        model_path = trainer.checkpoint_callback.last_model_path
    else:
        model_path = trainer.checkpoint_callback.best_model_path
    model = model_type.load_from_checkpoint(
        model_path, 
        num_classes_disease=num_classes_disease,
        inv_loss_coefficient=hparams.inv_loss_coefficient,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    def output_df(dataloader):
        info, embeddings, preds_disease, targets_disease, logits_disease, target_protected_attributes = test(model, dataloader, device)
    
        cols_names_classes_disease = ['class_' + str(i) for i in range(0, num_classes_disease)]
        cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
        cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

        df = pd.DataFrame(data=preds_disease, columns=cols_names_classes_disease)
        df_logits = pd.DataFrame(data=logits_disease, columns=cols_names_logits_disease)
        df_targets = pd.DataFrame(data=targets_disease, columns=cols_names_targets_disease)
        df_protected_attributes = pd.DataFrame(data=target_protected_attributes, columns=['Protected'])
        df = pd.concat([df, df_logits, df_targets, df_protected_attributes], axis=1)

        df_embed = pd.DataFrame(data=embeddings)
        df_embed = pd.concat([df_embed, df_targets, df_protected_attributes], axis=1)
        
        return info, df, df_embed

    print('VALIDATION')
    info, df_metrics, df_embedding = output_df(data_train.val_dataloader())
    with open(os.path.join(out_dir, 'count_info_val.json'), 'w') as f:
        json.dump(info, f, indent=4)
    df_metrics.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)
    df_embedding.to_csv(os.path.join(out_dir, 'embeddings.val.csv'), index=False)

    print('Train-Test')
    info, df_metrics, df_embedding = output_df(data_train.test_dataloader())
    with open(os.path.join(out_dir, 'count_info_test.json'), 'w') as f:
        json.dump(info, f, indent=4)
    df_metrics.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)
    df_embedding.to_csv(os.path.join(out_dir, 'embeddings.test.csv'), index=False)

    print('OOD Test')
    info, df_metrics, df_embedding = output_df(data_test.test_dataloader())
    with open(os.path.join(out_dir, 'count_info_ood.json'), 'w') as f:
        json.dump(info, f, indent=4)
    df_metrics.to_csv(os.path.join(out_dir, 'predictions.ood.csv'), index=False)
    df_embedding.to_csv(os.path.join(out_dir, 'embeddings.ood.csv'), index=False)


def cli():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--test', action="store_true")

    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--inv-loss-coefficient', type=int, default=1)
    parser.add_argument('--invariant-sampling', action='store_true')

    parser.add_argument('--protected-race-set-train', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--protected-race-set-test', nargs='+', type=int, default=[0, 1, 2, 3])

    parser.add_argument('--dataset-train', choices=["mimic", "chexpert"], default="chexpert")
    parser.add_argument('--dataset-test', choices=["mimic", "chexpert"], default="chexpert")
    parser.add_argument('--model-type', choices=["densenet", "resnet"], default="densenet")

    args = parser.parse_args()

    main(args)
