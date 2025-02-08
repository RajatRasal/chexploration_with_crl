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
from stocaching import SharedCache

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

from abc import ABC

load_dotenv()

image_size = (224, 224)
num_classes_disease = 14
num_classes_sex = 2
num_classes_race = 3
class_weights_race = (1.0, 1.0, 1.0)  # can be changed to balance accuracy
batch_size = 32
epochs = 20
num_workers = 4
img_data_dir = os.getenv("CHEXPERT_FOLDER")


class CheXpertDataset(Dataset):

    def __init__(
        self,
        csv_file_img,
        image_size, 
        augmentation = False, 
        pseudo_rgb = True,
        cache_size=0,
        nsamples=2,
        invariant_sampling = False,
    ):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.invariant_sampling = invariant_sampling
        self.use_cache = cache_size > 0

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices',
        ]

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        if self.invariant_sampling:
            protected_sex_attributes  = np.unique(self.data['sex_label'].values)
            protected_race_attributes = np.unique(self.data['race_label'].values)
            self.attribute_wise_samples = {}

            protected_sex_counts = np.zeros_like(protected_sex_attributes)
            protected_race_counts = np.zeros_like(protected_race_attributes)

            for i, sex in enumerate(protected_sex_attributes):
                self.attribute_wise_samples[sex] = {}
                protected_sex_counts[i] = np.sum(1. * (self.data['sex_label'].values == sex))

                for i, race in enumerate(protected_race_attributes):
                    self.attribute_wise_samples[sex][race] = {}
                    protected_race_counts[i] = np.sum(1. * (self.data['race_label'].values == race))


            protected_sex_counts = np.array(protected_sex_counts)
            protected_sex_probs  = 1. / protected_sex_counts

            protected_race_counts = np.array(protected_race_counts)
            protected_race_probs  = 1. / protected_race_counts

            self.nsamples = nsamples
            self.protected_race_probs = protected_race_probs / np.sum(protected_race_probs)
            self.protected_sex_probs  = protected_sex_probs / np.sum(protected_sex_probs)

            self.label_list = []

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label_disease = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label_disease[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            img_label_sex = np.array(self.data.loc[idx, 'sex_label'], dtype='int64')
            img_label_race = np.array(self.data.loc[idx, 'race_label'], dtype='int64')

            sample = {
                'image_path': img_path,
                'label_disease': img_label_disease,
                'label_sex': img_label_sex,
                'label_race': img_label_race,
            }
            self.samples.append(sample)

            if self.invariant_sampling:
                # invariant set selection -- update wrt task
                img_label_disease_ = [
                    img_label_disease[self.labels == 'No Finding'],
                    img_label_disease[self.labels == 'Pleural Effusion'],
                ]
                key = ','.join(map(str,img_label_disease_))
                if key not in self.attribute_wise_samples[int(img_label_sex)][int(img_label_race)]:
                    self.attribute_wise_samples[int(img_label_sex)][int(img_label_race)][key] = []

                self.attribute_wise_samples[int(img_label_sex)][int(img_label_race)][key].append(sample)

                if key not in self.label_list:
                    self.label_list.append(key)

        if self.use_cache:
            self.cache = {}

    def __len__(self):
        return len(self.data)

    def getitem_inv(self, item):
        samples = self.get_sample(item)

        images = []
        labels_disease = []
        labels_sex = []
        labels_race = []

        for sample in samples:
            image = sample['image'].unsqueeze(0)  # torch.from_numpy(sample['image']).unsqueeze(0)
            label_disease = torch.from_numpy(sample['label_disease'])
            label_sex = torch.from_numpy(sample['label_sex'])
            label_race = torch.from_numpy(sample['label_race'])

            if self.do_augment:
                image = self.augment(image)

            if self.pseudo_rgb:
                image = image.repeat(3, 1, 1)

            images.append(image.unsqueeze(0))
            labels_disease.append(label_disease.unsqueeze(0))
            labels_sex.append(label_sex.unsqueeze(0))
            labels_race.append(label_race.unsqueeze(0))

        images = torch.cat(images, 0)
        labels_disease = torch.cat(labels_disease, 0)
        labels_sex = torch.cat(labels_sex, 0)
        labels_race = torch.cat(labels_race, 0)

        return {
            'image': images,
            'label_disease': labels_disease, 
            'label_sex': labels_sex, 
            'label_race': labels_race,
        }

    def getitem(self, item):
        sample = self.get_sample(item)
        image = sample['image'].unsqueeze(0)  # torch.from_numpy(sample['image']).unsqueeze(0)
        label_disease = torch.from_numpy(sample['label_disease'])
        label_sex = torch.from_numpy(sample['label_sex'])
        label_race = torch.from_numpy(sample['label_race'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {
            'image': image, 
            'label_disease': label_disease, 
            'label_sex': label_sex, 
            'label_race': label_race,
        }

    def __getitem__(self, item):
        return self.getitem_inv(item) if self.invariant_sampling else self.getitem(item)
        
    def get_sample(self, item):
        if not self.invariant_sampling:
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
                'label_sex': sample['label_sex'], 
                'label_race': sample['label_race'],
            }
        return self.get_samples(item)

    def get_samples(self, item):
        np.random.seed(item)

        disease = np.random.choice(self.label_list)
        sex = np.random.choice(len(self.protected_sex_probs), self.nsamples, p=self.protected_sex_probs)
        race = np.random.choice(len(self.protected_race_probs), self.nsamples, p=self.protected_race_probs)
        
        info = []

        for si in range(self.nsamples):
            # enforce invariance in diseases
            sample = np.random.choice(self.attribute_wise_samples[sex[si]][race[si]][disease])

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
                'label_sex': sample['label_sex'], 
                'label_race': sample['label_race'],
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
        cache_size=0,
        nsamples=2,
        invariant_sampling=False,
    ):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nsamples = nsamples
        self.cache_size = cache_size
        self.invariant_sampling = invariant_sampling

        self.train_set = CheXpertDataset(
            self.csv_train_img, 
            self.image_size, 
            augmentation=True, 
            pseudo_rgb=pseudo_rgb,
            cache_size=self.cache_size,
            nsamples=self.nsamples,
            invariant_sampling=self.invariant_sampling,
        )
        self.val_set = CheXpertDataset(
            self.csv_val_img, 
            self.image_size, 
            augmentation=False, 
            pseudo_rgb=pseudo_rgb,
            nsamples=1,
            invariant_sampling=False,
        )
        self.test_set = CheXpertDataset(
            self.csv_test_img, 
            self.image_size, 
            augmentation=False, 
            pseudo_rgb=pseudo_rgb,
            nsamples=1,
            invariant_sampling=False,
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
        num_classes_sex, 
        num_classes_race, 
        class_weights_race,
    ):
        super().__init__()
        self.num_classes_disease = num_classes_disease
        self.num_classes_sex = num_classes_sex
        self.num_classes_race = num_classes_race
        self.class_weights_race = torch.FloatTensor(class_weights_race)

        self.fc_disease = None
        self.fc_sex = None
        self.fc_race = None
        self.backbone = None 

    def forward(self, x):
        embedding = self.backbone(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return embedding, out_disease, out_sex, out_race

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        embedding, out_disease, out_sex, out_race = self.forward(img)

        loss_disease = F.binary_cross_entropy_with_logits(out_disease, lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.to(img.device))
        loss_inv = 0

        return loss_disease, loss_sex, loss_race, loss_inv

    def process_batch_list(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        loss_inv = 0
        loss_disease = 0
        loss_sex = 0
        loss_race = 0
        prev_embedding = None

        for i in range(img.shape[0]):  
            embedding, out_disease, out_sex, out_race = self.forward(img[i])  
            loss_disease += F.binary_cross_entropy_with_logits(out_disease, lab_disease[i])
            loss_sex += F.cross_entropy(out_sex, lab_sex[i])
            loss_race += F.cross_entropy(
                out_race, lab_race[i], weight=self.class_weights_race.to(img.device),
            )
            
            if prev_embedding is not None:
                loss_inv += F.mse_loss(embedding, prev_embedding)
            prev_embedding = embedding

        return loss_disease, loss_sex, loss_race, loss_inv

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        params_sex = params_backbone + list(self.fc_sex.parameters())
        params_race = params_backbone + list(self.fc_race.parameters())
        optim_disease = torch.optim.Adam(params_disease, lr=0.001)
        optim_sex = torch.optim.Adam(params_sex, lr=0.001)
        optim_race = torch.optim.Adam(params_race, lr=0.001)
        return [optim_disease, optim_sex, optim_race]

    def training_step(self, batch, batch_idx):
        optim_disease, optim_sex, optim_race = self.optimizers()
        optim_disease.zero_grad()
        optim_sex.zero_grad()
        optim_race.zero_grad()

        invariant_rep = len(batch['image'].shape) == 5
        batch_processer = self.process_batch if not invariant_rep else self.process_batch_list

        loss_disease, loss_sex, loss_race, loss_inv = batch_processer(batch)
        self.log_dict({
            "train_loss_disease": loss_disease, 
            "train_loss_sex": loss_sex, 
            "train_loss_race": loss_race,
            "train_loss_inv": loss_inv,
        })

        samples = batch['image'] if not invariant_rep else batch['image'][0]

        grid = torchvision.utils.make_grid(samples[0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)

        self.manual_backward(loss_disease + loss_inv, retain_graph=True)
        self.manual_backward(loss_sex, retain_graph=True)
        self.manual_backward(loss_race)

        optim_disease.step()
        optim_sex.step()
        optim_race.step()

    def validation_step(self, batch, batch_idx):

        invariant_rep = len(batch['image'].shape) == 5
        batch_processer = self.process_batch if not invariant_rep else self.process_batch_list

        loss_disease, loss_sex, loss_race, loss_inv = batch_processer(batch)
        self.log_dict({
            "val_loss_disease": loss_disease, 
            "val_loss_sex": loss_sex, 
            "val_loss_race": loss_race,
            "val_loss_inv": loss_inv,
        })

    def test_step(self, batch, batch_idx):
        invariant_rep = len(batch['image'].shape) == 5
        batch_processer = self.process_batch if not invariant_rep else self.process_batch_list

        loss_disease, loss_sex, loss_race, loss_inv = batch_processer(batch)
        self.log_dict({
            "test_loss_disease": loss_disease, 
            "test_loss_sex": loss_sex, 
            "test_loss_race": loss_race,
            "test_loss_inv": loss_inv,
        })


class ResNet(BaseNet):

    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race):
        super().__init__(num_classes_disease, num_classes_sex, num_classes_race, class_weights_race)
        self.automatic_optimization = False  # Manual optimization needed
        self.backbone = models.resnet34(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.fc = self.fc_connect


class DenseNet(BaseNet):

    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race):
        super().__init__(num_classes_disease, num_classes_sex, num_classes_race, class_weights_race)
        self.automatic_optimization = False  # Manual optimization needed
        self.class_weights_race = torch.FloatTensor(class_weights_race)
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect


def test(model, data_loader, device):
    model.eval()
    logits_disease = []
    preds_disease = []
    targets_disease = []
    logits_sex = []
    preds_sex = []
    targets_sex = []
    logits_race = []
    preds_race = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            out_disease, out_sex, out_race = model(img)

            pred_disease = torch.sigmoid(out_disease)
            pred_sex = torch.softmax(out_sex, dim=1)
            pred_race = torch.softmax(out_race, dim=1)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            logits_sex.append(out_sex)
            preds_sex.append(pred_sex)
            targets_sex.append(lab_sex)

            logits_race.append(out_race)
            preds_race.append(pred_race)
            targets_race.append(lab_race)

        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)

        logits_sex = torch.cat(logits_sex, dim=0)
        preds_sex = torch.cat(preds_sex, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)

        logits_race = torch.cat(logits_race, dim=0)
        preds_race = torch.cat(preds_race, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

        counts = []
        for i in range(0,num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_sex):
            t = targets_sex == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_race):
            t = targets_race == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds_disease.cpu().numpy(), targets_disease.cpu().numpy(), logits_disease.cpu().numpy(), preds_sex.cpu().numpy(), targets_sex.cpu().numpy(), logits_sex.cpu().numpy(), preds_race.cpu().numpy(), targets_race.cpu().numpy(), logits_race.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets_disease = []
    targets_sex = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            emb = model.backbone(img)
            embeds.append(emb)
            targets_disease.append(lab_disease)
            targets_sex.append(lab_sex)
            targets_race.append(lab_race)

        embeds = torch.cat(embeds, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

    return embeds.cpu().numpy(), targets_disease.cpu().numpy(), targets_sex.cpu().numpy(), targets_race.cpu().numpy()


def main(hparams):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # data
    data = CheXpertDataModule(
        csv_train_img='datafiles/chexpert/chexpert.sample.train.csv',
        csv_val_img='datafiles/chexpert/chexpert.sample.val.csv',
        csv_test_img='datafiles/chexpert/chexpert.sample.test.csv',
        image_size=image_size,
        pseudo_rgb=True,
        batch_size=batch_size,
        num_workers=num_workers,
        nsamples=hparams.nsamples,
        invariant_sampling=hparams.invariant_sampling,
        cache_size=32,
    )

    # model
    model_type = DenseNet
    model = model_type(
        num_classes_disease=num_classes_disease,
        num_classes_sex=num_classes_sex,
        num_classes_race=num_classes_race,
        class_weights_race=class_weights_race,
    )

    # Create output directory
    out_name = 'densenet-all'
    if hparams.invariant_sampling:
        out_name = f'invariant-densenet-all-nsamples-{hparams.nsamples}'

    out_dir = 'chexpert/multitask/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.val_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].numpy().astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        max_epochs=epochs,
        devices=hparams.gpus,
        logger=TensorBoardLogger('chexpert/multitask', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    cols_names_classes_disease = ['class_' + str(i) for i in range(0,num_classes_disease)]
    cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
    cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

    cols_names_classes_sex = ['class_' + str(i) for i in range(0,num_classes_sex)]
    cols_names_logits_sex = ['logit_' + str(i) for i in range(0, num_classes_sex)]

    cols_names_classes_race = ['class_' + str(i) for i in range(0,num_classes_race)]
    cols_names_logits_race = ['logit_' + str(i) for i in range(0, num_classes_race)]

    print('VALIDATION')
    preds_val_disease, targets_val_disease, logits_val_disease, preds_val_sex, targets_val_sex, logits_val_sex, preds_val_race, targets_val_race, logits_val_race = test(model, data.val_dataloader(), device)
    
    df = pd.DataFrame(data=preds_val_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_val_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.disease.csv'), index=False)

    df = pd.DataFrame(data=preds_val_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_val_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_sex
    df.to_csv(os.path.join(out_dir, 'predictions.val.sex.csv'), index=False)

    df = pd.DataFrame(data=preds_val_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_val_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'predictions.val.race.csv'), index=False)

    print('TESTING')
    preds_test_disease, targets_test_disease, logits_test_disease, preds_test_sex, targets_test_sex, logits_test_sex, preds_test_race, targets_test_race, logits_test_race = test(model, data.test_dataloader(), device)
    
    df = pd.DataFrame(data=preds_test_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_test_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.disease.csv'), index=False)

    df = pd.DataFrame(data=preds_test_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_test_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_sex
    df.to_csv(os.path.join(out_dir, 'predictions.test.sex.csv'), index=False)

    df = pd.DataFrame(data=preds_test_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_test_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'predictions.test.race.csv'), index=False)

    print('EMBEDDINGS')
    embeds_val, targets_val_disease, targets_val_sex, targets_val_race = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets_disease = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_val_sex
    df['target_race'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'embeddings.val.csv'), index=False)

    embeds_test, targets_test_disease, targets_test_sex, targets_test_race = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets_disease = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_test_sex
    df['target_race'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'embeddings.test.csv'), index=False)


def cli():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type = int, default=1)
    parser.add_argument('--dev', type = int, default=0)

    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--invariant_sampling', action='store_true')
    args = parser.parse_args()

    main(args)
