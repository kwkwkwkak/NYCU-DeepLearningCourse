import os
import torch
import shutil
import numpy as np
import random
import cv2

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torchvision import transforms

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train"):

        assert mode in {"train", "valid", "test"}
    
        self.root = root
        self.mode = mode
        self.mean = [0.4783, 0.4463, 0.3961]
        self.std = [0.2631, 0.2575, 0.2655]
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        ])
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = Image.open(image_path).convert("RGB")

        if self.mode == "train" and random.random() > 0.5:
            image = self.color_jitter(image)
        
        image = np.array(image)

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap, filename=filename)

        sample['image'] = self.image_transform(sample['image'])
        sample['mask'] = self.mask_transform(sample['mask'])
        sample['trimap'] = self.mask_transform(sample['trimap'])

        if self.mode == "train": # only apply agumentation during training
            if random.random() > 0.5:
                sample['image'] = transforms.functional.hflip(sample['image'])
                sample['mask'] = transforms.functional.hflip(sample['mask'])
                sample['trimap'] = transforms.functional.hflip(sample['trimap'])
            
            if random.random() > 0.5:
                angle = 30 * (random.random() * 2 - 1)
                def random_rotate(img, interp=transforms.InterpolationMode.BILINEAR, resize_interp=transforms.InterpolationMode.BILINEAR):
                    upscaled = transforms.functional.resize(img, (336, 336), interpolation=resize_interp)
                    return transforms.functional.center_crop(
                        transforms.functional.rotate(upscaled, 
                                                     angle, 
                                                     interpolation=interp), 
                        (256, 256)
                    )
                
                sample['image'] = random_rotate(sample['image'])
                sample['mask'] = random_rotate(sample['mask'], interp=transforms.InterpolationMode.NEAREST, resize_interp=transforms.InterpolationMode.NEAREST_EXACT)
                sample['trimap'] = random_rotate(sample['trimap'], interp=transforms.InterpolationMode.NEAREST, resize_interp=transforms.InterpolationMode.NEAREST_EXACT)
                
            if random.random() > 0.5:
                top, left = random.randint(0, 32), random.randint(0, 32)

                def random_crop(img, interp=transforms.InterpolationMode.BILINEAR):
                    return transforms.functional.resize(
                        transforms.functional.crop(img, top, left, 224, 224),
                        (256, 256),
                        interpolation=interp
                    )
                
                sample['image'] = random_crop(sample['image'])
                sample['mask'] = random_crop(sample['mask'], transforms.InterpolationMode.NEAREST_EXACT)
                sample['trimap'] = random_crop(sample['trimap'], transforms.InterpolationMode.NEAREST_EXACT)

            if random.random() > 0.7:
                sample['image'] = transforms.functional.gaussian_blur(sample['image'], kernel_size=3)
            
            if random.random() > 0.7:
                noise = torch.randn_like(sample['image']) * 0.07
                sample['image'] = torch.clamp(sample['image'] + noise, 0, 1)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        
        blacklist = ['saint_bernard_108', 'Egyptian_Mau_162', 'Egyptian_Mau_196', 'Egyptian_Mau_165', 'leonberger_18',
                    'miniature_pinscher_14', 'saint_bernard_15']
        
        filenames = [x.split(" ")[0] for x in split_data if x.split(" ")[0] not in blacklist]

        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR), dtype=np.float32)
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST), dtype=np.float32)
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST), dtype=np.float32)

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    # implement the load dataset function here
    
    return OxfordPetDataset(data_path, mode)

if __name__ == "__main__":
    loader = load_dataset(os.path.join(os.getcwd(), '..\\dataset\\oxford-iiit-pet\\'), mode='train')

    from torch.utils.data import DataLoader
    dataloader = DataLoader(loader, batch_size=1, shuffle=False, num_workers=4)

    sum_rgb = torch.zeros(3)
    sum_sq_rgb = torch.zeros(3)
    num_pixels = 0

    for batch in dataloader:
        image = batch['image']  # 1, c, h, w
        cv2.imshow("asd", np.flip(image[0].numpy().transpose(1, 2, 0), axis=2))
        cv2.waitKey()
        # sum_rgb += image.sum(dim=[0, 2, 3]) 
        # sum_sq_rgb += (image ** 2).sum(dim=[0, 2, 3])  
        
        # num_pixels += 256 ** 2  

    # mean = sum_rgb / num_pixels
    # std = torch.sqrt((sum_sq_rgb / num_pixels) - (mean ** 2))

    # print(f"mean: {mean}")
    # print(f"std: {std}")