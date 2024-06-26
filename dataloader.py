import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.transform = transform
    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

                
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = image/80.0 ## (512, 512, 1)
        image = np.expand_dims(image, axis=0) ## (1, 512, 512)
        image = image.astype(np.float32)
        

        mask = mask/255.0   ## (512, 512, 1)
        mask[mask>=0.5]= 1
        mask[mask<0.5] = 0
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)

        if self.transform is None:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples