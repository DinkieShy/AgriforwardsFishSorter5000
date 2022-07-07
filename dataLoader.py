from random import random as rand
import os
from torchvision.io import read_image
from torchvision.transforms import RandomPerspective

class DataLoader(directory):
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
            for filename in os.listdir(self.directory + className):
                if os.isfile(filename):
                    self.images.append(filename)
                    self.labels.append(className)

    def __len__(self):
        return len(self.images)

    def __getItem__(index):
        image_id = self.images[index]
        label = self.labels[index]

        image = read_image(image_id)

        if self.augment:
            if rand() > 0.5:
                image = self.randomPerspective.forward(image)

        return image, self.labelEnum[label]