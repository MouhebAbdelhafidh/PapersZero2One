from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from utils import get_image_paths_from_dir
from PIL import Image

class DatasetConfig:
    def __init__(self, dataset_path="train_split", image_size=64, flip=True, to_normal=True):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.flip = flip
        self.to_normal = to_normal

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal

    def __len__(self):
        return self._length * 2 if self.flip else self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = transform(image)
        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
        return image

class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(f"{dataset_config.dataset_path}/B")
        image_paths_cond = get_image_paths_from_dir(f"{dataset_config.dataset_path}/A")
        self.flip = dataset_config.flip if stage=='train' else False
        self.to_normal = dataset_config.to_normal
        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]
