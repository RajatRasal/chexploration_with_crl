import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
from skimage.io import imread

import os
import cv2
import numbers
import torchvision.transforms.functional as TF

from sklearn.utils import shuffle
from skimage.transform import resize
from skimage.util import img_as_ubyte
from sampler import SamplerFactory


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

            protected_race_probs = 1. / protected_race_counts

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

        if self.use_cache:
            self.cache = {}

    def __len__(self):
        return int(len(self.samples)/self.nsamples)

    def getitem_inv(self, item):
        samples = self.get_sample(item)

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
                'label': sample['label'], 
                'invariant_attribute': sample['invariant_attribute'],
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
                'label': sample['label'], 
                'label': sample['invariant_attribute'],
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
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

# ======================================================

class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""

    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, "gammacorrection")

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: gamma corrected image.
        """
        gamma_factor = (
            None
            if self.gamma is None
            else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        )
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img


class MammoDataset(Dataset):
    def __init__(
        self,
        data,
        image_size,
        image_normalization,
        horizontal_flip=False,
        augmentation=False,
        use_cache=False,
        nsamples=2,
        invariant_sampling=False,
        attribute_set=['mlo', 'cc']
    ):
        self.image_size = image_size
        self.image_normalization = image_normalization
        self.do_flip = horizontal_flip
        self.do_augment = augmentation
        self.use_cache = use_cache
        self.attribute_set = attribute_set
        self.invariant_sampling = invariant_sampling
        self.nsamples = nsamples


        # photometric data augmentation
        self.photometric_augment = T.Compose(
            [
                GammaCorrectionTransform(gamma=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

        # geometric data augmentation
        self.geometric_augment = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    transforms=[T.RandomAffine(degrees=10, scale=(0.95, 1.05))], p=0.75
                ),
            ]
        )

        self.img_paths = data.img_path.to_numpy()
        self.study_ids = data.study_id.to_numpy()
        self.image_ids = data.image_id.to_numpy()

        self.views = data.view_position.to_numpy()
        self.density = data.density_label.to_numpy()
        

        self.attribute_wise_samples = {'mlo':{}, 'cc':{}}
        self.samples = []
        self.unique_densities = []
        for idx, _ in enumerate(tqdm(range(len(self.img_paths)), desc='Loading Data')):
            sample = {
                'image_path': self.img_paths[idx],
                'study_ids': self.study_ids[idx],
                'image_ids': self.image_ids[idx],
                'view': self.views[idx],
                'density': self.density[idx]
            }

            if sample['view'] not in self.attribute_set:
                continue
                
            if density not in self.attribute_wise_samples[sample['view']]:
                self.attribute_wise_samples[sample['view']][density] = []
                self.unique_densities.append(density)

            self.attribute_wise_samples[sample['view']][sample['density']].append(sample)
            self.samples.append(sample)


        # initialize the cache
        if self.use_cache:
            self.cache = {}


    def preprocess(self, image, horizontal_flip):
        # resample
        if self.image_size != image.shape:
            image = resize(image, output_shape=self.image_size, preserve_range=True)

        # breast mask
        image_norm = image - np.min(image)
        image_norm = image_norm / np.max(image_norm)
        thresh = cv2.threshold(img_as_ubyte(image_norm), 5, 255, cv2.THRESH_BINARY)[
            1
        ]

        # Connected components with stats.
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
            thresh, connectivity=4
        )

        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        max_label, _ = max(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
            key=lambda x: x[1],
        )
        mask = output == max_label
        image[mask == 0] = 0

        # flip
        if horizontal_flip:
            left = np.mean(image[:, 0 : int(image.shape[1] / 2)])  # noqa
            right = np.mean(image[:, int(image.shape[1] / 2) : :])  # noqa
            if left < right:
                image = image[:, ::-1].copy()

        return image

    def __len__(self):
        return int(len(self.samples)/self.nsamples)

    def __getitem__(self, item):
        return self.getitem_inv(item) if self.invariant_sampling else self.getitem(item)
    
            
    def get_image(self, img_path):
        image = None
        if self.use_cache:
            if img_path in self.cache.keys():
                image = self.cache[img_path]

        if image is None:
            image = imread(img_path).astype(np.float32)
            horizontal_flip = self.do_flip
            image = self.preprocess(image, horizontal_flip)
            image = torch.from_numpy(image).unsqueeze(0)

            if self.use_cache:
                self.cache[img_path] = image

        # normalize intensities to range [0,1]
        image = image / self.image_normalization

        if self.do_augment:
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)

        image = image.repeat(3, 1, 1)
        return image


    def getitem_inv(self, index):
        np.random.seed(index)

        # Sample a disease
        density = np.random.choice(self.unique_densities)
        views = np.random.choice(
            self.attribute_wise_samples.keys(),
            self.nsamples
        )

        # Get samples for the chosen disease from the race invariant set.
        # This allows ensures that representations for the same disease
        # are invariant to race when trained in a fashion akin to https://arxiv.org/abs/2106.04619.
        info = []
        for si in range(self.nsamples):
            sample = np.random.choice(self.attribute_wise_samples[views[si]][density])

            image = self.get_image(sample['image_path'])
                
            info.append({
                'image': image, 
                'label': sample['density'], 
                'label': sample['view'],
            })  
        return info

    def getitem(self, index):
        sample = self.samples[index]

        image = self.get_image(sample['image_path'])

        return {
            "image": image,
            "label": self.labels[index],
            "study_id": self.study_ids[index],
            "image_id": self.img_paths[index],
        }

    def get_labels(self):
        return self.labels


class EMBEDMammoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_file,
        image_size,
        data_dir,
        batch_alpha=0,
        batch_size=32,
        num_workers=6,
        split_dataset=True,
        nsamples=2,
        invariant_sampling=False,
        use_cache=False,
        view_set_train=['mlo', 'cc'],
        view_set_test=['mlo', 'cc'],
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_alpha = batch_alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.invariant_sampling = invariant_sampling
        self.nsamples = nsamples 
        self.use_cache = use_cache
        self.view_set_test = view_set_test
        self.view_set_train = view_set_train

        if isinstance(csv_file, pd.DataFrame):
            self.data = csv_file
        else:
            self.data = pd.read_csv(csv_file)

        test_percent = 0.25
        val_percent = 0.1
        # FFDM only
        self.data = self.data[self.data["FinalImageType"] == "2D"]

        # Female only
        self.data = self.data[self.data["GENDER_DESC"] == "Female"]

        # Remove unclear breast density cases
        self.data = self.data[self.data["tissueden"].notna()]
        self.data = self.data[self.data["tissueden"] < 5]

        # MLO and CC only
        self.data = self.data[self.data["ViewPosition"].isin(["MLO", "CC"])]

        # Remove spot compression or magnificiation
        self.data = self.data[self.data["spot_mag"].isna()]
        self.data["laterality"] = self.data["ImageLateralityFinal"]

        self.test_percent = test_percent
        self.val_percent = val_percent

        self.data["img_path"] = [
            os.path.join(self.data_dir, img_path)
            for img_path in self.data.image_path.values
        ]
        self.data["study_id"] = [
            str(study_id) for study_id in self.data.empi_anon.values
        ]
        self.data["image_id"] = [
            img_path.split("/")[-1] for img_path in self.data.image_path.values
        ]

        self.data["view_position"] = [
            str(view).lower() for view in self.data.ViewPosition.values
        ]

        # Define density categories
        self.data["density_label"] = 0
        self.data.loc[self.data["tissueden"] == 1, "density_label"] = 0
        self.data.loc[self.data["tissueden"] == 2, "density_label"] = 1
        self.data.loc[self.data["tissueden"] == 3, "density_label"] = 2
        self.data.loc[self.data["tissueden"] == 4, "density_label"] = 3

        # Split data into training, validation, and testing
        # Making sure images from the same subject are within the same set
        self.data["split"] = "test"
        unique_study_ids_all = self.data.empi_anon.unique()
        unique_study_ids_all = shuffle(unique_study_ids_all, random_state=33)
        num_test = round(len(unique_study_ids_all) * self.test_percent)

        dev_sub_id = unique_study_ids_all[num_test:]
        self.data.loc[self.data.empi_anon.isin(dev_sub_id), "split"] = "training"

        self.dev_data = self.data[self.data["split"] == "training"]
        self.test_data = self.data[self.data["split"] == "test"]

        unique_study_ids_dev = self.dev_data.empi_anon.unique()

        unique_study_ids_dev = shuffle(unique_study_ids_dev, random_state=33)
        num_train = round(len(unique_study_ids_dev) * (1.0 - self.val_percent))

        valid_sub_id = unique_study_ids_dev[num_train:]
        self.dev_data.loc[self.dev_data.empi_anon.isin(valid_sub_id), "split"] = (
            "validation"
        )

        self.train_data = self.dev_data[self.dev_data["split"] == "training"]
        self.val_data = self.dev_data[self.dev_data["split"] == "validation"]

        self.train_set = MammoDataset(
            data=self.train_data,
            image_size=self.image_size,
            image_normalization=65535.0,
            horizontal_flip=True,
            augmentation=True,
            use_cache=self.use_cache,
            nsamples=self.nsamples,
            invariant_sampling=self.invariant_sampling,
            attribute_set=self.view_set_train
        )
        self.val_set = MammoDataset(
            data=self.val_data,
            image_size=self.image_size,
            image_normalization=65535.0,
            horizontal_flip=True,
            augmentation=False,
            use_cache=self.use_cache,
            nsamples=self.nsamples,
            invariant_sampling=self.invariant_sampling,
            attribute_set=self.view_set_test
        )
        self.test_set = MammoDataset(
            data=self.test_data,
            image_size=self.image_size,
            image_normalization=65535.0,
            horizontal_flip=True,
            augmentation=False,
            use_cache=self.use_cache,
            nsamples=self.nsamples,
            invariant_sampling=self.invariant_sampling,
            attribute_set=self.view_set_test
        )

        train_labels = self.train_set.get_labels()
        val_labels = self.val_set.get_labels()
        test_labels = self.test_set.get_labels()

        if self.batch_alpha > 0:
            train_class_idx = [
                np.where(train_labels == t)[0] for t in np.unique(train_labels)
            ]
            train_batches = len(self.train_set) // self.batch_size

            self.train_sampler = SamplerFactory().get(
                train_class_idx,
                self.batch_size,
                train_batches,
                alpha=self.batch_alpha,
                kind="fixed",
            )

        print("samples (train): ", len(self.train_set))
        print("samples (val):   ", len(self.val_set))
        print("samples (test):  ", len(self.test_set))

          


    def train_dataloader(self):
        if self.batch_alpha == 0:
            return DataLoader(
                dataset=self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            return DataLoader(
                dataset=self.train_set,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
