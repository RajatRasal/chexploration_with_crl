import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
from skimage.io import imread


class CXRDataset(Dataset):

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
        self.nsamples = nsamples

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

            # TODO: to make sure it is similar to non-invariant training
            # protected_race_probs = 1. / protected_race_counts

            self.protected_race_probs = protected_race_counts / np.sum(protected_race_counts)

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
                'label': img_label_disease,
                'invariant_attribute': img_label_race,
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

        if self.invariant_sampling:
            self.label_count = {}
            print(self.label_list)
            print(self.attribute_wise_samples.keys())
            for race in self.attribute_wise_samples.keys():
                for disease in self.attribute_wise_samples[race].keys():
                    print(race, disease, len(self.attribute_wise_samples[race][disease]))
                    self.label_count[disease] = len(self.attribute_wise_samples[race][disease])

        if self.use_cache:
            self.cache = {}

    def __len__(self):
        return int(len(self.samples))

    def getitem_inv(self, item):
        samples = self.get_samples(item)

        images = []
        labels_disease = []
        labels_race = []

        for sample in samples:
            image = sample['image'].unsqueeze(0)
            label_disease = torch.tensor(sample['label'])
            label_race = torch.tensor(sample['invariant_attribute'])

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
            'label': labels_disease, 
            'invariant_attribute': labels_race,
        }

    def getitem(self, item):
        sample = self.get_sample(item)
        image = sample['image'].unsqueeze(0)
        label_disease = torch.tensor(sample['label'])
        label_race = torch.tensor(sample['invariant_attribute'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {
            'image': image, 
            'label': label_disease, 
            'invariant_attribute': label_race,
        }

    def __getitem__(self, item):
        return self.getitem_inv(item) if self.invariant_sampling else self.getitem(item)
        
    def get_sample(self, item):
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
            'label': sample['label'], 
            'invariant_attribute': sample['invariant_attribute'],
        }

    def get_samples(self, item):
        np.random.seed(item)

        # Sample a disease
        # disease = np.random.choice(
        #     np.array(list(self.label_count.keys())),
        #     1,
        #     p=np.array(list(self.label_count.values())) / np.sum(np.array(list(self.label_count.values())))
        # )[0]
        diseases = np.array(list(self.label_count.keys()))
        disease = np.random.choice(diseases, size=1)[0]
        prob = self.protected_race_probs[self.protected_race_set]
        prob = prob / np.sum(prob)  # renormalising
        race = np.random.choice(
            self.protected_race_set,
            self.nsamples,
            # p=prob,
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
                'label': sample['label'], 
                'invariant_attribute': sample['invariant_attribute'],
            })  
        return info

    def _read_image(self, image_path):
        return torch.from_numpy(imread(image_path).astype(np.float32))


class CXRDataModule(pl.LightningDataModule):

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

        self.train_set = CXRDataset(
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
        self.val_set = CXRDataset(
            self.csv_val_img, 
            self.image_size, 
            img_data_dir,
            augmentation=False, 
            pseudo_rgb=pseudo_rgb,
            nsamples=1,
            invariant_sampling=False,
            protected_race_set=self.protected_race_set_train,
        )
        self.test_set = CXRDataset(
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
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

