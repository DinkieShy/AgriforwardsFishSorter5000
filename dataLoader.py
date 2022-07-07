from random import random as rand
import os
from torchvision.io import read_image
from torchvision.transforms import RandomPerspective
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, directory, augment=False):
        self.directory = directory
        self.augment = augment
        self.images = []
        self.labels = []
        self.labelEnum = {}
        self.dataframe = []
        self.randomPerspective = RandomPerspective()

        for className in os.listdir(self.directory):
            self.labelEnum[className] = len(self.labelEnum)
            for filename in os.listdir(self.directory + "/" + className):
                if os.path.isfile(self.directory + "/" + className + "/" + filename):
                    self.images.append(self.directory + "/" + className + "/" + filename)
                    self.labels.append(className)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
		assert index < len(self.images), "index error when accessing dataset"

        image_id = self.images[index]
        label = self.labels[index]

        image = read_image(image_id)

        if self.augment:
            if rand() > 0.5:
                image = self.randomPerspective.forward(image)

        return image, self.labelEnum[label]