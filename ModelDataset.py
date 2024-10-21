import os
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root_dir: str, label_dir: str, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx: int):
        image_name = self.image_path[idx]
        image_item_path = os.path.join(self.path, image_name)
        image = Image.open(image_item_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.label_dir in ["Fake", "fake"]:
            label = 0
        elif self.label_dir in ["Real", "real"]:
            label = 1
        else:
            label = 2
        return image, label

    def __len__(self):
        return len(self.image_path)


