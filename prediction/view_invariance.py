import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from dotenv import load_dotenv
from typing import Literal
import torchvision
from tqdm import tqdm
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imsave
from argparse import ArgumentParser

import json

from prediction.backbones import DenseNet, ResNet34, ViTB16
from prediction.datasets.embed import EMBEDMammoDataModule, VINDRMammoDataModule
from prediction.metrics import compute_metrics


load_dotenv()

num_classes_density = 4
batch_size = 32
num_workers = 4


def datafiles(dataset: Literal["embed", "vindr"]):
    files = []
    if dataset == "embed":
        if Path(os.getenv("EMBED_FOLDER_CLUSTER")).exists():
            print("USING CLUSTER")
            files.append(os.getenv("EMBED_FOLDER_CLUSTER"))
        else:
            print("USING LOCAL")
            files.append(os.getenv("EMBED_FOLDER"))
        files.append("/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv")
    else:
        # TODO: Implement and integreate vindr dataset
        files.append(os.getenv("VINDR_FOLDER"))
        files.append(os.path.join(os.getenv("VINDR_FOLDER"), "breast-level_annotations.csv"))
        # "/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv")
    return files


def test(model, data_loader, device, test_run):
    model.eval()
    logits_density = []
    preds_density = []
    embeddings = []
    targets_density = []
    targets_invariant_attributes = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            if test_run and index == 3:
                break

            img, label, invariant_attribute = batch['image'].to(device), batch['label'].to(device), batch['invariant_attribute'].to(device)
            embedding, out_density = model(img)

            pred = torch.softmax(out_density, dim=-1)

            logits_density.append(out_density)
            preds_density.append(torch.argmax(pred, -1))
            targets_density.append(label)

            embeddings.append(embedding)
            targets_invariant_attributes.append(invariant_attribute)

        embeddings = torch.cat(embeddings, dim=0)
        logits_density = torch.cat(logits_density, dim=0)
        preds_density = torch.cat(preds_density, dim=0)
        targets_density = torch.cat(targets_density, dim=0)
        targets_invariant_attributes = torch.cat(targets_invariant_attributes, dim=0)

        info = {'global': {}, 'sub-group': {}}

        for i in range(0, num_classes_density):
            t = targets_density == i
            c = torch.sum(t).item()
            info['global'][i] = int(c)

        for j in torch.unique(targets_invariant_attributes):
            info['sub-group'][int(j.item())] = {}
            for i in range(0, num_classes_density):
                t = targets_density[targets_invariant_attributes == j] == i
                c = torch.sum(t).item()
                info['sub-group'][int(j.item())][i] = int(c)

        print(json.dumps(info, indent=4))

    return (
        info,
        embeddings.cpu().numpy(),
        preds_density.cpu().numpy(),
        targets_density.cpu().numpy(),
        logits_density.cpu().numpy(),
        targets_invariant_attributes.cpu().numpy(),
    )


def main(hparams):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)

    # model
    model_type = {
        "densenet": DenseNet,
        "resnet34": ResNet34,
        "vitb16": ViTB16,
    }[hparams.model_type]

    model = model_type(
        num_classes=num_classes_density,
        inv_loss_coefficient=hparams.inv_loss_coefficient,
    )

    image_size = (hparams.size, hparams.size)

    # Create output directory
    logdir = "logs_test" if hparams.test else "logs"
    logdir = f'{logdir}/view_invariance_{hparams.size}'

    # Setup datasets
    # Train set
    data_dir, csv_file = datafiles(hparams.dataset_train)
    # Select datamodule
    dms = {"embed": EMBEDMammoDataModule, "vindr": VINDRMammoDataModule}
    if hparams.dataset_train == hparams.dataset_test:
        # Attribute_transfer - train with protected_race_set_train test with protected_race_set_test
        out_dir = f'{logdir}/{model_type.__name__}-{hparams.seed}/{hparams.dataset_train}/{hparams.view_set_train}_{hparams.view_set_test}'
       
        data_train = dms[hparams.dataset_train](
            csv_file=csv_file,
            image_size=image_size,
            data_dir=data_dir,
            batch_alpha=0,
            batch_size=batch_size,
            num_workers=num_workers,
            split_dataset=True,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            view_set_train=hparams.view_set_train,
            view_set_test=hparams.view_set_train,
        )
        data_test = dms[hparams.dataset_test](
            csv_file=csv_file,
            image_size=image_size,
            data_dir=data_dir,
            batch_alpha=0,
            batch_size=batch_size,
            num_workers=num_workers,
            split_dataset=True,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            view_set_train=hparams.view_set_train,
            view_set_test=hparams.view_set_test,
        )
    else:
        # Dataset transfer - train with data_train, test on data_test
        out_dir = f'{logdir}/{model_type.__name__}-{hparams.seed}/{hparams.dataset_train}_{hparams.dataset_test}'

        print(csv_file, data_dir)
        data_train = dms[hparams.dataset_train](
            csv_file=csv_file,
            image_size=image_size,
            data_dir=data_dir,
            batch_alpha=0,
            batch_size=batch_size,
            num_workers=num_workers,
            split_dataset=True,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            view_set_train=hparams.view_set_train,
            view_set_test=hparams.view_set_test,
        )

        # Alternative test set 
        data_dir, csv_file = datafiles(hparams.dataset_test)
        data_test = dms[hparams.dataset_test](
            csv_file=csv_file,
            image_size=image_size,
            data_dir=data_dir,
            batch_alpha=0,
            batch_size=batch_size,
            num_workers=num_workers,
            split_dataset=True,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            view_set_train=hparams.view_set_train,
            view_set_test=hparams.view_set_test,
        )

    # Set up logging dirs
    if hparams.invariant_sampling:
        out_dir = os.path.join(out_dir, f"invariant_nsamples_{hparams.nsamples}")
    else:
        out_dir = os.path.join(out_dir, "non_invariant")
    os.makedirs(out_dir, exist_ok=True)

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_class",
        mode="min",
        save_last=True,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        max_epochs=hparams.epochs,
        devices=hparams.gpus,
        logger=TensorBoardLogger(out_dir),
        max_steps=5 if hparams.test else -1,
    )
    trainer.logger._default_hp_metric = False
    log_dir = trainer.log_dir

    # Sample images
    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    for idx in range(0, 5):
        sample = data_train.val_set.get_sample(idx)
        imsave(
            os.path.join(samples_dir, 'sample_' + str(idx) + '.png'),
            (255 * sample['image']).numpy().astype(np.uint8),
        )

    # Train
    trainer.fit(model, data_train)

    # Test
    if hparams.test:
        model_path = trainer.checkpoint_callback.last_model_path
    else:
        model_path = trainer.checkpoint_callback.best_model_path
    model = model_type.load_from_checkpoint(
        model_path, 
        num_classes=num_classes_density,
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
            preds_density,
            targets_density,
            logits_density,
            target_invariant_attributes,
        ) = test(model, dataloader, device, hparams.test)
    
        cols_names_logits_density = ['logit_' + str(i) for i in range(0, num_classes_density)]

        df = pd.DataFrame(data=preds_density, columns=['class'])
        df_logits = pd.DataFrame(data=logits_density, columns=cols_names_logits_density)
        df_targets = pd.DataFrame(data=targets_density, columns=['target'])
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
        metrics_dir = os.path.join(log_dir, "output")
        os.makedirs(metrics_dir, exist_ok=True)
        dl = dls[split]
        info, df_preds, df_embedding, df_metrics = output_dfs(dl)
        with open(os.path.join(metrics_dir, f'count_info_{split}.json'), 'w') as f:
            json.dump(info, f, indent=4)
        df_preds.to_csv(os.path.join(metrics_dir, f'predictions.{split}.csv'), index=False)
        df_embedding.to_csv(os.path.join(metrics_dir, f'embeddings.{split}.csv'), index=False)
        df_metrics.to_csv(os.path.join(metrics_dir, f'metrics.{split}.csv'), index=False)

    write_dfs("val")
    write_dfs("test")
    write_dfs("ood")


def cli():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--test', action="store_true")

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--inv-loss-coefficient', type=int, default=1)
    parser.add_argument('--invariant-sampling', action='store_true')

    parser.add_argument('--view-set-train', nargs='+', type=str, default=['mlo', 'cc'])
    parser.add_argument('--view-set-test', nargs='+', type=str, default=['mlo', 'cc'])

    parser.add_argument('--dataset-train', choices=["embed", "vindr"], default="embed")
    parser.add_argument('--dataset-test', choices=["embed", "vindr"], default="embed")
    parser.add_argument('--model-type', choices=["densenet", "resnet34", "vitb16"], default="densenet")

    args = parser.parse_args()

    main(args)
