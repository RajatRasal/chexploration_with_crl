import os

import pandas as pd
import torch

from prediction.chexpert_multitask import (
    DenseNet,
    num_classes_disease,
    num_classes_sex,
    num_classes_race,
    class_weights_race,
    test,
    image_size,
    batch_size,
    num_workers,
    CheXpertDataModule,
    embeddings,
)


def main():
    data = CheXpertDataModule(
        csv_train_img='datafiles/chexpert/chexpert.sample.train.csv',
        csv_val_img='datafiles/chexpert/chexpert.sample.val.csv',
        csv_test_img='datafiles/chexpert/chexpert.sample.test.csv',
        image_size=image_size,
        pseudo_rgb=True,
        batch_size=batch_size,
        num_workers=num_workers,
        nsamples=1, # hparams.nsamples,
        invariant_sampling=False, # hparams.invariant_sampling,
        use_cache=True,
    )

    # model_name = 'densenet-all'
    # os.path.join(out_dir, "version_26/checkpoints/epoch=4-step=28800.ckpt"),
    # model_name = 'invariant-densenet-all-nsamples-2-10'
    # os.path.join(out_dir, "version_1/checkpoints/epoch=3-step=21600.ckpt"),
    model_name = 'invariant-densenet-all-nsamples-2-50'
    out_dir = 'chexpert/multitask/' + model_name
    path = os.path.join(out_dir, "version_1/checkpoints/epoch=4-step=28800.ckpt")

    model = DenseNet.load_from_checkpoint(
        path,
        num_classes_disease=num_classes_disease,
        num_classes_sex=num_classes_sex,
        num_classes_race=num_classes_race,
        class_weights_race=class_weights_race,
        inv_loss_coefficient=0,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model.to(device)

    cols_names_classes_disease = ['class_' + str(i) for i in range(0, num_classes_disease)]
    cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
    cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

    cols_names_classes_sex = ['class_' + str(i) for i in range(0, num_classes_sex)]
    cols_names_logits_sex = ['logit_' + str(i) for i in range(0, num_classes_sex)]

    cols_names_classes_race = ['class_' + str(i) for i in range(0, num_classes_race)]
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


if __name__ == "__main__":
    main()