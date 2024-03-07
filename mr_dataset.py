import logging
import os
import cv2
import pickle
import random
import numbers
import numpy as np
from math import pi
from pathlib import Path
from collections.abc import Sequence
from typing import Optional, Union, Tuple

import cfl
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torchvision import transforms
from torchvision.utils import _log_api_usage_once


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0], size[0]
    if len(size) != 3:
        raise ValueError(error_msg)
    return size


class RandomCrop3D(torch.nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int, int]):# -> Tuple[int, int, int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, k, h, w, d) to be passed to ``crop`` for random crop.
        """
        _, h, w, d = img.shape[-4:]
        th, tw, td = output_size

        if h + 1 < th or w + 1 < tw or d + 1 < td:
            raise ValueError(f"Required crop size {(th, tw, td)} is larger then input image size {(h, w, d)}")

        if w == tw and h == th and d == td:
            return 0, 0, h, w, d

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        k = torch.randint(0, d - td + 1, size=(1,)).item()
        return i, j, k, th, tw, td

    def __init__(self, size):
        super().__init__()
        _log_api_usage_once(self)
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w, d) for size."))

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        i, j, k, h, w, d = self.get_params(img, self.size)
        bottom = i + h
        right = j + w
        back = k + d
        return img[..., i:bottom, j:right, k:back]


class ToTensor(object):
    """  Array to Tensor (2D)  """
    def __init__(self):
        pass
        
    def __call__(self, images):
        if isinstance(images, dict):
            image = images['img']
            target = images['target']
            if image.ndim == 3: # 2D image
                sample = {'img': torch.Tensor(image).permute(2, 0, 1), 'target': torch.Tensor(target).permute(2, 0, 1)}
                return sample
            elif image.ndim == 4: # 3D image
                sample = {'img': torch.Tensor(image).permute(3, 0, 1, 2), 'target': torch.Tensor(target).permute(3, 0, 1, 2)}
                return sample
        
        elif images.ndim == 3: # 2D image
            return torch.Tensor(images).permute(2, 0, 1)
        elif images.ndim == 4: # 3D image
            return torch.Tensor(images).permute(3, 0, 1, 2)


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    Input Image: Load Complex 1 channel array and convert it to Real/Imaginary/Magnitude/Phase 4 channel array
    Target Image: Load Complex 1 channel array and convert it to Magnitude/Phase 2 channel array
    """

    def __init__(
        self,
        train_path: Union[str, Path, os.PathLike],
        sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        is_train: bool = True,
        args: Optional[str] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
        """
        self.dataset_cache_file = Path(dataset_cache_file)

        self.examples = []
        self.is_train = is_train
        self.input_size = args.input_size
        self.crop = args.crop
        self.crop_eval = args.crop_eval
        self.input_type = args.input_type
        self.output_type = args.output_type
        self.eval_target = args.eval_target
        
        folder_name = 'train' if self.is_train else 'val'
        root = os.path.join(train_path, folder_name)

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            dirs = sorted(list(Path(root).iterdir()))
            if self.eval_target:
                inputs = dirs[1]
            else:
                inputs = dirs[0]
            self.examples += [fname for fname in list(Path(inputs).iterdir())]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        if self.is_train:
            pass
        else:
            self.examples = sorted(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname = self.examples[i]
            
        if os.path.splitext(fname)[-1] in ['.npz', '.npy']:
            img = np.load(fname)
        elif os.path.splitext(fname)[-1] in ['.cfl', '.hdr']:
            img = np.squeeze(cfl.read_cfl(fname[:-4]))
            
        img_list = []
        if self.input_type == 'RI':
            img_list.append(img.real)
            img_list.append(img.imag)
        elif self.input_type == 'MP':
            img_list.append(np.abs(img))
            img_list.append(np.angle(img) / pi)
        elif self.input_type == 'M':
            img_list.append(np.abs(img))
        elif self.input_type == 'RIMP':
            img_list.append(img.real)
            img_list.append(img.imag)
            img_list.append(np.abs(img))
            img_list.append(np.angle(img) / pi)
        elif self.input_type == 'MMM':
            img_list.append(np.abs(img))
            img_list.append(np.abs(img))
            img_list.append(np.abs(img))
        else:
            raise Exception('Invalid choice of input image type!')
        img = np.stack(img_list, axis=-1)
        
        target = str(fname).replace('input', 'target')
        target = np.load(target)
        
        target_list = []
        if self.output_type == 'RI':
            target_list.append(target.real)
            target_list.append(target.imag)
        elif self.output_type == 'MP':
            target_list.append(np.abs(target))
            target_list.append(np.angle(target) / pi)
        elif self.output_type == 'M':
            target_list.append(np.abs(target))
        elif self.output_type == 'RIMP':
            target_list.append(target.real)
            target_list.append(target.imag)
            target_list.append(np.abs(target))
            target_list.append(np.angle(target) / pi)
        elif self.output_type == 'MMM':
            target_list.append(np.abs(target))
            target_list.append(np.abs(target))
            target_list.append(np.abs(target))
        else:
            raise Exception('Invalid choice of output/target image type!')
        target = np.stack(target_list, axis=-1)

        sample = {'img': img, 'target': target}
        sample = self._transform(sample)

        return sample
    
    def _transform(self, sample):
        if isinstance(sample, dict):
            image = torch.Tensor(sample['img']).permute(2, 0, 1)
            target = torch.Tensor(sample['target']).permute(2, 0, 1)
            
            if self.is_train:
                if self.crop:
                    # Crop
                    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.input_size, self.input_size))
                    image = TF.crop(image, i, j, h, w)
                    target = TF.crop(target, i, j, h, w)
                
                # Rotate (90, 180, 270) and vertical flip
                image, target = self._geometric_augmentation([image, target])
                return image, target
            else:
                if self.crop_eval:
                    # Crop center
                    image = TF.center_crop(image, output_size=(self.input_size, self.input_size))
                    target = TF.center_crop(target, output_size=(self.input_size, self.input_size))
                else:
                    image = image
                    target = target
                return image, target
        else:
            assert False, 'Use torchvision.transform!'
            
    def _geometric_augmentation(self, sample):
        flag_aug = torch.randint(0, 8, (1,))
        
        if flag_aug[0] == 0:
            out = sample
        elif flag_aug[0] == 1:
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        elif flag_aug[0] == 2:
            sample[0] = torch.rot90(sample[0], 1, [1, 2])
            sample[1] = torch.rot90(sample[1], 1, [1, 2])
            out = sample
        elif flag_aug[0] == 3:
            sample[0] = torch.rot90(sample[0], 1, [1, 2])
            sample[1] = torch.rot90(sample[1], 1, [1, 2])
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        elif flag_aug[0] == 4:
            sample[0] = torch.rot90(sample[0], 2, [1, 2])
            sample[1] = torch.rot90(sample[1], 2, [1, 2])
            out = sample
        elif flag_aug[0] == 5:
            sample[0] = torch.rot90(sample[0], 2, [1, 2])
            sample[1] = torch.rot90(sample[1], 2, [1, 2])
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        elif flag_aug[0] == 6:
            sample[0] = torch.rot90(sample[0], 3, [1, 2])
            sample[1] = torch.rot90(sample[1], 3, [1, 2])
            out = sample
        elif flag_aug[0] == 7:
            sample[0] = torch.rot90(sample[0], 3, [1, 2])
            sample[1] = torch.rot90(sample[1], 3, [1, 2])
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        else:
            raise Exception('Invalid choice of image transformation')
        return out

class RGBDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to RGB image slices.
    Input Image: Load RGB 3 channel array
    Target Image: Load RGB 3 channel array
    """

    def __init__(
        self,
        train_path: Union[str, Path, os.PathLike],
        sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        is_train: bool = True,
        args: Optional[str] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
        """
        self.dataset_cache_file = Path(dataset_cache_file)

        self.examples = []
        self.is_train = is_train
        self.input_size = args.input_size
        self.crop = args.crop
        self.crop_eval = args.crop_eval
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        folder_name = 'train' if self.is_train else 'val'
        root = os.path.join(train_path, folder_name)

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            dirs = sorted(list(Path(root).iterdir()))
            inputs = dirs[0]
            self.examples += [fname for fname in list(Path(inputs).iterdir())]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        if self.is_train:
            pass
        else:
            self.examples = sorted(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname = str(self.examples[i])
            
        if os.path.splitext(fname)[-1] in ['.png', '.jpg']:
            img = cv2.imread(fname) / 255.
        else:
            assert FileNotFoundError('Wrong file extension!')
                
        target_fname = str(fname).replace('input', 'target')
        if os.path.splitext(target_fname)[-1] in ['.png', '.jpg']:
            target = cv2.imread(target_fname) / 255.
        else:
            assert FileNotFoundError('Wrong file extension!')
            
        sample = {'img': img, 'target': target}
        sample = self._transform(sample)

        return sample
    
    def _transform(self, sample):
        if isinstance(sample, dict):
            image = torch.Tensor(sample['img']).permute(2, 0, 1)
            target = torch.Tensor(sample['target']).permute(2, 0, 1)
            
            if self.is_train:
                if self.crop:
                    # Crop
                    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.input_size, self.input_size))
                    image = TF.crop(image, i, j, h, w)
                    target = TF.crop(target, i, j, h, w)
                
                # Rotate (90, 180, 270) and vertical flip
                image, target = self._geometric_augmentation([image, target])
                return image, target
            else:
                if self.crop_eval:
                    # Crop center
                    image = TF.center_crop(image, output_size=(self.input_size, self.input_size))
                    target = TF.center_crop(target, output_size=(self.input_size, self.input_size))
                else:
                    image = image
                    target = target
                return image, target
        else:
            assert False, 'Use torchvision.transform!'
            
    def _geometric_augmentation(self, sample):
        flag_aug = torch.randint(0, 8, (1,))
        
        if flag_aug[0] == 0:
            out = sample
        elif flag_aug[0] == 1:
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        elif flag_aug[0] == 2:
            sample[0] = torch.rot90(sample[0], 1, [1, 2])
            sample[1] = torch.rot90(sample[1], 1, [1, 2])
            out = sample
        elif flag_aug[0] == 3:
            sample[0] = torch.rot90(sample[0], 1, [1, 2])
            sample[1] = torch.rot90(sample[1], 1, [1, 2])
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        elif flag_aug[0] == 4:
            sample[0] = torch.rot90(sample[0], 2, [1, 2])
            sample[1] = torch.rot90(sample[1], 2, [1, 2])
            out = sample
        elif flag_aug[0] == 5:
            sample[0] = torch.rot90(sample[0], 2, [1, 2])
            sample[1] = torch.rot90(sample[1], 2, [1, 2])
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        elif flag_aug[0] == 6:
            sample[0] = torch.rot90(sample[0], 3, [1, 2])
            sample[1] = torch.rot90(sample[1], 3, [1, 2])
            out = sample
        elif flag_aug[0] == 7:
            sample[0] = torch.rot90(sample[0], 3, [1, 2])
            sample[1] = torch.rot90(sample[1], 3, [1, 2])
            sample[0] = torch.flip(sample[0], [1, 2])
            sample[1] = torch.flip(sample[1], [1, 2])
            out = sample
        else:
            raise Exception('Invalid choice of image transformation')
        return out


class CubeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image cubes.
    """
    def __init__(
        self,
        train_path: Union[str, Path, os.PathLike],
        sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        is_train: bool = True,
        args: Optional[str] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
        """
        self.dataset_cache_file = Path(dataset_cache_file)

        self.examples = []
        self.is_train = is_train
        self.input_size = args.input_size
        self.crop = args.crop
        self.crop_eval = args.crop_eval
        
        folder_name = 'train' if self.is_train else 'val'
        root = os.path.join(train_path, folder_name)

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            dirs = sorted(list(Path(root).iterdir()))
            inputs = dirs[0]
            self.examples += [fname for fname in list(Path(inputs).iterdir())]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        if self.is_train:
            pass
        else:
            self.examples = sorted(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname = self.examples[i]
            
        if os.path.splitext(fname)[-1] in ['.npz', '.npy']:
            img = np.load(fname)
        elif os.path.splitext(fname)[-1] in ['.cfl', '.hdr']:
            img = np.squeeze(cfl.read_cfl(fname[:-4]))
        img = np.stack((img.real, img.imag), axis=-1)
        
        target = str(fname).replace('input', 'target')
        target = np.load(target)
        target = np.stack((target.real, target.imag), axis=-1)#.transpose(1, 2, 0)

        sample = {'img': img, 'target': target}
        sample = self._transform(sample)

        return sample
    
    def _transform(self, sample):
        if isinstance(sample, dict):
            image = torch.Tensor(sample['img']).permute(3, 0, 1, 2)
            target = torch.Tensor(sample['target']).permute(3, 0, 1, 2)
            
            if self.is_train:
                if self.crop:
                    # Random Crop
                    i, j, k, h, w, d = RandomCrop3D.get_params(image, output_size=(self.input_size, self.input_size, self.input_size))
                    image = image[..., i:i + h, j:j + w, k:k + d]
                    target = target[..., i:i + h, j:j + w, k:k + d]
                # Rotate (90, 180, 270) and vertical flip
                image, target = self._geometric_augmentation([image, target])
                
                return image, target
            else:
                if self.crop_eval:
                    # Crop center
                    image = image[:, image.size()[1]//2-64:image.size()[1]//2+64, 
                                  image.size()[2]//2-112:image.size()[2]//2+112, 
                                  image.size()[3]//2-112:image.size()[3]//2+112]
                    target = target[:, target.size()[1]//2-64:target.size()[1]//2+64, 
                                  target.size()[2]//2-112:target.size()[2]//2+112, 
                                  target.size()[3]//2-112:target.size()[3]//2+112]
                else:
                    image = image#[:, z_center-80:z_center+80, ...]
                    target = target#[:, z_center-80:z_center+80, ...]
                return image, target
        else:
            assert False, 'Use torchvision.transform!'
            
    def _geometric_augmentation(self, sample):
        flag_aug = torch.randint(0, 8, (1,))
        flag_aug2 = torch.randint(2, 3, (1,))
        
        if flag_aug[0] == 0:
            out = sample
        elif flag_aug[0] == 1:
            if flag_aug2[0] == 0:
                sample[0] = torch.flip(sample[0], [1, 2])
                sample[1] = torch.flip(sample[1], [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.flip(sample[0], [1, 3])
                sample[1] = torch.flip(sample[1], [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.flip(sample[0], [2, 3])
                sample[1] = torch.flip(sample[1], [2, 3])
            out = sample
        elif flag_aug[0] == 2:
            if flag_aug2[0] == 0:
                sample[0] = torch.rot90(sample[0], 1, [1, 2])
                sample[1] = torch.rot90(sample[1], 1, [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.rot90(sample[0], 1, [1, 3])
                sample[1] = torch.rot90(sample[1], 1, [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.rot90(sample[0], 1, [2, 3])
                sample[1] = torch.rot90(sample[1], 1, [2, 3])
            out = sample
        elif flag_aug[0] == 3:
            if flag_aug2[0] == 0:
                sample[0] = torch.rot90(sample[0], 1, [1, 2])
                sample[1] = torch.rot90(sample[1], 1, [1, 2])
                sample[0] = torch.flip(sample[0], [1, 2])
                sample[1] = torch.flip(sample[1], [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.rot90(sample[0], 1, [1, 3])
                sample[1] = torch.rot90(sample[1], 1, [1, 3])
                sample[0] = torch.flip(sample[0], [1, 3])
                sample[1] = torch.flip(sample[1], [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.rot90(sample[0], 1, [2, 3])
                sample[1] = torch.rot90(sample[1], 1, [2, 3])
                sample[0] = torch.flip(sample[0], [2, 3])
                sample[1] = torch.flip(sample[1], [2, 3])
            out = sample
        elif flag_aug[0] == 4:
            if flag_aug2[0] == 0:
                sample[0] = torch.rot90(sample[0], 2, [1, 2])
                sample[1] = torch.rot90(sample[1], 2, [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.rot90(sample[0], 2, [1, 3])
                sample[1] = torch.rot90(sample[1], 2, [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.rot90(sample[0], 2, [2, 3])
                sample[1] = torch.rot90(sample[1], 2, [2, 3])
            out = sample
        elif flag_aug[0] == 5:
            if flag_aug2[0] == 0:
                sample[0] = torch.rot90(sample[0], 2, [1, 2])
                sample[1] = torch.rot90(sample[1], 2, [1, 2])
                sample[0] = torch.flip(sample[0], [1, 2])
                sample[1] = torch.flip(sample[1], [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.rot90(sample[0], 2, [1, 3])
                sample[1] = torch.rot90(sample[1], 2, [1, 3])
                sample[0] = torch.flip(sample[0], [1, 3])
                sample[1] = torch.flip(sample[1], [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.rot90(sample[0], 2, [2, 3])
                sample[1] = torch.rot90(sample[1], 2, [2, 3])
                sample[0] = torch.flip(sample[0], [2, 3])
                sample[1] = torch.flip(sample[1], [2, 3])
            out = sample
        elif flag_aug[0] == 6:
            if flag_aug2[0] == 0:
                sample[0] = torch.rot90(sample[0], 3, [1, 2])
                sample[1] = torch.rot90(sample[1], 3, [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.rot90(sample[0], 3, [1, 3])
                sample[1] = torch.rot90(sample[1], 3, [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.rot90(sample[0], 3, [2, 3])
                sample[1] = torch.rot90(sample[1], 3, [2, 3])
            out = sample
        elif flag_aug[0] == 7:
            if flag_aug2[0] == 0:
                sample[0] = torch.rot90(sample[0], 3, [1, 2])
                sample[1] = torch.rot90(sample[1], 3, [1, 2])
                sample[0] = torch.flip(sample[0], [1, 2])
                sample[1] = torch.flip(sample[1], [1, 2])
            elif flag_aug2[0] == 1:
                sample[0] = torch.rot90(sample[0], 3, [1, 3])
                sample[1] = torch.rot90(sample[1], 3, [1, 3])
                sample[0] = torch.flip(sample[0], [1, 3])
                sample[1] = torch.flip(sample[1], [1, 3])
            elif flag_aug2[0] == 2:
                sample[0] = torch.rot90(sample[0], 3, [2, 3])
                sample[1] = torch.rot90(sample[1], 3, [2, 3])
                sample[0] = torch.flip(sample[0], [2, 3])
                sample[1] = torch.flip(sample[1], [2, 3])
            out = sample
        else:
            raise Exception('Invalid choice of image transformation')
        return out
        
def build_dataset(is_train, args):
    if args.data_set == 'MRI2D':
        dataset = SliceDataset(args.data_path, is_train=is_train, args=args)
        return dataset, None
    elif args.data_set == 'MRI3D':
        dataset = CubeDataset(args.data_path, is_train=is_train, args=args)
        return dataset, None
    else:
        dataset = RGBDataset(args.data_path, is_train=is_train, args=args)
        return dataset, None
