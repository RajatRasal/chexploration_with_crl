import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from dotenv import load_dotenv
from typing import Literal
import torchvision
from tqdm import tqdm

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imsave
from argparse import ArgumentParser

import json

from prediction.backbones import DenseNet, ResNet, ViTB16
from prediction.datasets.cxr import CXRDataModule, CXRDataset
from prediction.metrics import compute_metrics


load_dotenv()

image_size = (224, 224)
num_classes_disease = 2
batch_size = 32
epochs = 25
num_workers = 4


def datafiles(dataset: Literal["chexpert", "mimic"]):
    files = [
        f"datafiles/{dataset}/{dataset}.sample.{split}.csv"
        for split in ["train", "val", "test"]
    ]
    if dataset == "chexpert":
        files.append(os.getenv("CHEXPERT_FOLDER"))
    else:
        files.append(os.getenv("MIMIC_FOLDER"))
    return files


def test(model, data_loader, device, test_run):
    model.eval()
    logits_disease = []
    preds_disease = []
    embeddings = []
    targets_disease = []
    targets_invariant_attributes = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            if test_run and index == 3:
                break

            img, lab_disease, invariant_attribute = batch['image'].to(device), batch['label'].to(device), batch['invariant_attribute'].to(device)
            embedding, out_disease = model(img)

            pred_disease = torch.sigmoid(out_disease)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            embeddings.append(embedding)
            targets_invariant_attributes.append(invariant_attribute)

        embeddings = torch.cat(embeddings, dim=0)
        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_invariant_attributes = torch.cat(targets_invariant_attributes, dim=0)

        info = {'global': {}, 'sub-group': {}}

        for i in range(0, num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t).item()
            info['global'][i] = int(c)

        for j in torch.unique(targets_invariant_attributes):
            info['sub-group'][int(j.item())] = {}
            for i in range(0, num_classes_disease):
                t = targets_disease[targets_invariant_attributes == j] == i
                c = torch.sum(t).item()
                info['sub-group'][int(j.item())][i] = int(c)

        print(json.dumps(info, indent=4))

    return (
        info,
        embeddings.cpu().numpy(),
        preds_disease.cpu().numpy(),
        targets_disease.cpu().numpy(),
        logits_disease.cpu().numpy(),
        targets_invariant_attributes.cpu().numpy(),
    )


def main(hparams):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)

    # model
    model_type = {
        "densenet": DenseNet,
        "resnet": ResNet,
        "vitb16": ViTB16,
    }[hparams.model_type]

    model = model_type(
        num_classes_disease=num_classes_disease,
        inv_loss_coefficient=hparams.inv_loss_coefficient,
    )

    # Create output directory
    logdir = "logs_test" if hparams.test else "logs"
    logdir = f'{logdir}/race_invariance'
    # Setup datasets
    # Train set
    (
        csv_train_img,
        csv_val_img,
        csv_test_img,
        img_data_dir,
    ) = datafiles(hparams.dataset_train)
    if hparams.dataset_train == hparams.dataset_test:
        # Attribute_transfer - train with protected_race_set_train test with protected_race_set_test
        out_dir = f'{logdir}/{model_type.__name__}-{hparams.seed}/{hparams.dataset_train}/{hparams.protected_race_set_train}_{hparams.protected_race_set_test}'
       
        data_train = CXRDataModule(
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
        data_test = CXRDataModule(
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
        # Dataset transfer - train with data_train, test on data_test
        out_dir = f'{logdir}/{model_type.__name__}-{hparams.seed}/{hparams.dataset_train}_{hparams.dataset_test}'
       
        data_train = CXRDataModule(
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

        # Alternative test set 
        (
            test_csv_train_img,
            test_csv_val_img,
            test_csv_test_img,
            test_img_data_dir,
        ) = datafiles(hparams.dataset_test)
        data_test = CXRDataModule(
            csv_train_img=test_csv_train_img,
            csv_val_img=test_csv_val_img,
            csv_test_img=test_csv_test_img,
            image_size=image_size,
            img_data_dir=test_img_data_dir,
            pseudo_rgb=True,
            batch_size=batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test,
        )

    # Set up logging dirs
    if hparams.invariant_sampling:
        out_dir = os.path.join(out_dir, f"invariant_nsamples_{hparams.nsamples}")
    else:
        out_dir = os.path.join(out_dir, "non_invariant")
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Sample images
    for idx in range(0, 5):
        sample = data_train.val_set.get_sample(idx)
        imsave(
            os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'),
            sample['image'].numpy().astype(np.uint8),
        )

    # Train
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_class",
        mode="min",
        save_last=True,
    )
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

    # Test
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

    # Compute metrics and embeddings and write to CSV
    def output_dfs(dataloader):
        (
            info,
            embeddings,
            preds_disease,
            targets_disease,
            logits_disease,
            target_invariant_attributes,
        ) = test(model, dataloader, device, hparams.test)
    
        cols_names_classes_disease = ['class_' + str(i) for i in range(0, num_classes_disease)]
        cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
        cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

        df = pd.DataFrame(data=preds_disease, columns=cols_names_classes_disease)
        df_logits = pd.DataFrame(data=logits_disease, columns=cols_names_logits_disease)
        df_targets = pd.DataFrame(data=targets_disease, columns=cols_names_targets_disease)
        df_invariant_attributes = pd.DataFrame(data=target_invariant_attributes, columns=['Protected'])
        df_preds = pd.concat([df, df_logits, df_targets, df_invariant_attributes], axis=1)

        df_metrics = compute_metrics(df_preds)

        df_embed = pd.DataFrame(data=embeddings)
        df_embed = pd.concat([df_embed, df_targets, df_invariant_attributes], axis=1)
        
        return info, df_preds, df_embed, df_metrics

    dls = {
        "val": data_train.val_dataloader(),
        "test": data_train.test_dataloader(),
        "ood": data_test.test_dataloader(),
    }
    def write_dfs(split: Literal["val", "test", "ood"]):
        dl = dls[split]
        info, df_preds, df_embedding, df_metrics = output_dfs(dl)
        with open(os.path.join(out_dir, f'count_info_{split}.json'), 'w') as f:
            json.dump(info, f, indent=4)
        df_preds.to_csv(os.path.join(out_dir, f'predictions.{split}.csv'), index=False)
        df_embedding.to_csv(os.path.join(out_dir, f'embeddings.{split}.csv'), index=False)
        df_metrics.to_csv(os.path.join(out_dir, f'metrics.{split}.csv'), index=False)

    write_dfs("val")
    write_dfs("test")
    write_dfs("ood")


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
    parser.add_argument('--model-type', choices=["densenet", "resnet", "vitb16"], default="densenet")

    args = parser.parse_args()

    main(args)
