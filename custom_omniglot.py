from torch.utils.data import Dataset
import os
import pickle
import torch
import torchvision.transforms as T
from PIL import Image


class CustomOmniglot(Dataset):
    def __init__(self, examples_dir):

        self.examples_dir = examples_dir

        header_path = os.path.join(examples_dir, 'header.pkl')
        header_obj = pickle.load(open(header_path, "rb"))
        self.len = header_obj['len']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        img_dir = str(idx)
        transform = T.PILToTensor()

        img1_file = Image.open(os.path.join(self.examples_dir, img_dir, 'img1.png'))
        img2_file = Image.open(os.path.join(self.examples_dir, img_dir, 'img2.png'))

        img1_tensor = transform(img1_file)
        img2_tensor = transform(img2_file)

        # X: tensor with a dimension added for each image
        img_shape = list(img1_tensor.shape)

        x = torch.empty([2] + img_shape)
        x[0] = img1_tensor
        x[1] = img2_tensor

        with open(os.path.join(self.examples_dir, img_dir, 'y.pkl'), 'rb') as y_file:
            y = pickle.load(y_file)

        return x, y

class OmniglotLRMatch(Dataset):
    def __init__(self, examples_dir):

        self.examples_dir = examples_dir

        header_path = os.path.join(examples_dir, 'header.pkl')
        header_obj = pickle.load(open(header_path, "rb"))
        self.len = header_obj['len']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        img_dir = str(idx)
        transform = T.PILToTensor()

        img1_file = Image.open(os.path.join(self.examples_dir, img_dir, 'img1.png'))
        img2_file = Image.open(os.path.join(self.examples_dir, img_dir, 'img2.png'))
        img3_file = Image.open(os.path.join(self.examples_dir, img_dir, 'img3.png'))

        img1_tensor = transform(img1_file)
        img2_tensor = transform(img2_file)
        img3_tensor = transform(img3_file)

        # X: tensor with a dimension added for each image
        img_shape = list(img1_tensor.shape)

        x = torch.empty([3] + img_shape)
        x[0] = img1_tensor
        x[1] = img2_tensor
        x[2] = img3_tensor

        with open(os.path.join(self.examples_dir, img_dir, 'y.pkl'), 'rb') as y_file:
            y = pickle.load(y_file)

        return x, y