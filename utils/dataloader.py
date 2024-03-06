import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import numpy as np


class SegmentationDataset(Dataset):
    """Класс для загрузки изображений и масок с аугментацией"""

    def __init__(self, images: list, masks: list, image_size: tuple, augment=True):
        """Class for loading images and detection labels
        and converting them to segmentation masks

        Parameters
        ----------
        images : list
            List of image pathes
        masks : list
            List of mask pathes
        image_size : tuple
            Input size of model (height, width)
        augment : bool, optional
            Whether to use augmentation, by default True
        """
        self.images = images
        self.masks = masks
        self.augment = augment
        self.image_size = image_size
        affine_aug = A.OneOf([
              A.Rotate(60, p=0.25),
              A.Flip(p=0.25),
              A.Affine(scale=(0.5, 0.5), p=0.25),
              A.MaskDropout(5, p=0.25)
        ], p=0.5)
        self.transform = A.Compose([
              affine_aug,
              A.OneOf([
                    A.MotionBlur(p=0.125),
                    A.OpticalDistortion(p=0.125),
                    A.ISONoise(p=0.125),
                    A.GaussNoise(p=0.125),
                    A.RandomFog(0.2, 0.6, p=0.125),
                    A.RandomRain(blur_value=2, p=0.125),
                    A.RandomShadow(p=0.125),
                    A.RandomBrightnessContrast(p=0.125)
              ], p=0.5)
        ], p=0.5)


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB) # open image
        image = cv2.resize(image, self.image_size)  # resize image
        mask = cv2.imread(self.masks[idx], 0)  # open mask in grayscale
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)  # resize image
        if self.augment:
              augmented = self.transform(image=image, mask=mask)
              image = augmented['image']
              mask = augmented['mask']
        mask = torch.from_numpy(mask / 255).int()  # convert to tensor
        image = torch.from_numpy(image / 255.0).float()

        return torch.permute(image, (2, 0, 1)), torch.unsqueeze(mask, dim=0)
    



class DetectionToSegmentationDataset(Dataset):
    """Class to convert detection txt labels to
    segmentation masks
    """

    def __init__(self, images: list, labels: list, image_size: tuple, augment=True):
        """Class for loading images and detection labels
        and converting them to segmentation masks

        Parameters
        ----------
        images : list
            List of image pathes
        labels : list
            List of label pathes
        image_size : tuple
            Input size of model (height, width)
        augment : bool, optional
            Whether to use augmentation, by default True
        """
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.ratio = np.array((image_size[1], image_size[0]))
        self.augment = augment
        affine_aug = A.OneOf([
              A.Rotate(60, p=0.25),
              A.Flip(p=0.25),
              A.Affine(scale=(0.5, 0.5), p=0.25),
              A.MaskDropout(5, p=0.25)
        ], p=0.5)
        self.transform = A.Compose([
              affine_aug,
              A.OneOf([
                    A.MotionBlur(p=0.125),
                    A.OpticalDistortion(p=0.125),
                    A.ISONoise(p=0.125),
                    A.GaussNoise(p=0.125),
                    A.RandomFog(0.2, 0.6, p=0.125),
                    A.RandomRain(blur_value=2, p=0.125),
                    A.RandomShadow(p=0.125),
                    A.RandomBrightnessContrast(p=0.125)
              ], p=0.5)
        ], p=0.5)


    def __len__(self):
        return len(self.images)
    

    def __convert_labels(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes to segmentation masks

        Parameters
        ----------
        boxes : np.ndarray
            Bounding boxes in YOLO format (class_id, x, y, width, height)

        Returns
        -------
        np.ndarray
            Segmentation mask
        """
        mask = np.zeros(self.image_size, int)  # init mask
        boxes = np.concatenate((boxes[:, 1:3] * self.ratio, boxes[:, 3:] * self.ratio), 
                               axis=1).astype(int)  # scale boxes
        for box in boxes:
            x, y, w, h = box  # get object coordinates
            mask[y - h//2 : y + h//2, x - w//2 : x + w//2] = 1  # draw 1 to object
        return mask
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB) # open image
        image = cv2.resize(image, self.image_size)  # resize image
        with open(self.labels[idx], 'r') as f:  # read labels from txt
            labels = f.readlines()
        labels = [[float(x) for x in l.split()] for l in labels]  # split string and convert to float
        labels = np.array(labels)
        mask = self.__convert_labels(labels)
        # perform augmentation
        if self.augment:
              augmented = self.transform(image=image, mask=mask)
              image = augmented['image']
              mask = augmented['mask']
        mask = torch.from_numpy(mask).int()  # convert to tensor
        image = torch.from_numpy(image / 255.0).float()

        return torch.permute(image, (2, 0, 1)), torch.unsqueeze(mask, dim=0)