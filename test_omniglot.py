import enum
import json
import pickle
import matplotlib.pyplot as plt
import os
from sympy import I
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomImageDataset
import torch
from torch import nn, optim
import statistics
from torchvision import datasets, transforms
import random
import torchvision.transforms as T
from torchvision.utils import save_image


from shape_mem_encoder import MemoryEncoder
from functional import FunctionalNN

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

MEMORY_SIZE = 128

MEMORY_LENGTH = 2

device = 'cuda'

if __name__ == '__main__':

    functional_source = 'functionalnn.pkl'
    functional = pickle.load(open(functional_source, 'rb'))

    mem_encoder_source = 'encoder-.011.pkl'
    mem_encoder = pickle.load(open(mem_encoder_source, 'rb'))

    # NOTE: ALL DATA IS VALIDATE

    # Invert: because omniglot data is black on white, but we want
    # white on black

    all_transforms = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2])),
                    transforms.RandomInvert(p=1)])

    data = datasets.Omniglot('omniglot', download=True, transform=all_transforms)
    loader = DataLoader(data, batch_size=len(data), shuffle=True)

    new_data = []

    # Create new dataset based on omniglot to perform memory task
    # NOTE: NOT QUITE DONE YET

    for batch, (X, y) in enumerate(loader):

        # Expand into three color channels
        X = X.repeat((1, INPUT_SHAPE[0] // X.shape[1], 1, 1))

        # Sort characters into type
        label_dict = {}  # Label --> Image tensor
        label_index = {}  # Dict to keep track of something later
        for i in range(len(X)):
            if y[i].item() not in label_dict:
                label_dict[y[i].item()] = []
                label_index[y[i].item()] = 0
            label_dict[y[i].item()].append(X[i])

        # print(list(label_index.keys()))
        examples = []

        my_keys = list(label_index.keys())
        for k in my_keys:
            while label_index[k] < len(label_dict[k]):
                new_tensors = []
                new_y = torch.zeros((2,))
                # Half get same type; half different type (roughly)
                if random.random() < .5:
                    if len(label_dict[k]) > label_index[k] + 1:
                        new_tensors.append(label_dict[k][label_index[k]])
                        new_tensors.append(label_dict[k][label_index[k]+1])
                        label_index[k] += 2
                        new_y[0] = 1.0
                    else:
                        break
                else:
                    new_tensors.append(label_dict[k][label_index[k]])
                    available_others = []  # list of available other category types
                    for label in label_index:
                        if label != k and len(label_dict[label]) > label_index[label]:
                            available_others.append(label)
                    if len(available_others) == 0:
                        break
                    randint = random.randrange(0, len(available_others))
                    chosen = available_others[randint]
                    new_tensors.append(label_dict[chosen][label_index[chosen]])
                    label_index[available_others[randint]] += 1
                    label_index[k] += 1
                    new_y[1] = 1.0

                new_tensors = tuple(new_tensors)
                examples.append((new_tensors, new_y))
        new_data.extend(examples)


    """counts = 0
    for (X, y) in new_data:
        if y[0] == 0.:
            counts += 1
    print(counts)
    print(len(new_data))

    counter = 0
    random.shuffle(new_data)
    for (X, y) in new_data:
        plt.imshow(X[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(X[1].permute(1, 2, 0))
        plt.show()
        print(y)

        if counter == 5:
            break
        counter += 1"""

    # WRITE OUT DATASET

    print("Finished prep.")
    print("Len: " + str(len(new_data)))

    directory = 'new-omni'
    header_filename = 'header.pkl'
    trans = T.ToPILImage()
    for i, (X, y) in enumerate(new_data):
        try:
            os.mkdir(os.path.join(directory, str(i)))
        except FileExistsError:
            pass

        save_image(X[0], os.path.join(directory, str(i), 'img1.png'))
        save_image(X[1], os.path.join(directory, str(i), 'img2.png'))

        fhand = open(os.path.join(directory, str(i), 'y.pkl'), 'wb')
        pickle.dump(y, fhand)
        fhand.close()
        if i % 50 == 0:
            print(f"{i} done.")

    header_file = open(os.path.join(directory, header_filename), 'wb')
    pickle.dump({'len': len(new_data), 'desc': 'X contains tensors of two images. y is tensor [1.0, 0.] if they\'re the same type. Otherwise [0., 1.]'}, header_file)
    print("Done.")

