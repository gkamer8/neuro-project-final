import matplotlib.pyplot as plt
from sympy import false, true
import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomImageDataset, LongMatchOrNoGame
import pickle
from custom_omniglot import CustomOmniglot

"""

Used to produce visualizations for the paper

"""

def show_dataset(dataset_src):
    # Show shape dataset
    # You're expected to manually save the pictures you want as pngs

    TO_SHOW = 1

    dataset = LongMatchOrNoGame(dataset_src)
    loader = DataLoader(dataset, batch_size=TO_SHOW, shuffle=True)

    # Just look at one batch
    for (X, y) in loader:
        for i in range(TO_SHOW):
            img1 = X[i][0]
            img2 = X[i][1]
            img3 = X[i][2]

            plt.imshow(img1.detach().cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(img2.detach().cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(img3.detach().cpu().permute(1, 2, 0))
            plt.show()
        break


def plot_encoder_training():

    SHOW_LRS = true

    train_info_file = "encoder_train_info.pkl"
    info = pickle.load(open(train_info_file, "rb"))

    y = info['val_losses']
    x = [i+1 for i in range(len(y))]

    # only look at first _ epochs
    crop_to_first = 100
    x = x[:crop_to_first]
    y = y[:crop_to_first]
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plt.plot(x, y)
    plt.title("Validation Loss of Memory Autoencoder")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.show()

def plot_recognizer_training():

    SHOW_LRS = false

    train_info_file = "recognizer_train_info_sornot_noise_try2.pkl"
    info = pickle.load(open(train_info_file, "rb"))

    y = info['val_accuracies']
    x = [i+1 for i in range(len(y))]

    # only look at first _ epochs
    crop_to_first = 50
    x = x[:crop_to_first]
    y = y[:crop_to_first]
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plt.plot(x, y)
    plt.title("Validation Accuracy of Game 2 with Noise")
    plt.xlabel("Examples Shown (10,000s)")
    plt.ylabel("Accuracy %")
    plt.show()

def plot_multiple():

    SHOW_LRS = false

    # only look at first _ epochs
    crop_to_first = 60

    """filenames = [
        'visuals/check-in/noise-lrmatch/recognizer_train_info_noise.pkl',
        'visuals/check-in/rotation-lrmatch/recognizer_train_info_rot.pkl',
        'visuals/check-in/displacement-lrmatch/recognizer_train_info_displacement.pkl'
        ]"""
    filenames = [
        'visuals/check-in/noise-sornot2/recognizer_train_info_sornot_noise_try2.pkl',
        'visuals/check-in/rotation-sornot2/recognizer_train_info_sornot_rotation_small.pkl',
        'visuals/check-in/displacement-sornot2/recognizer_train_info_sornot_displacement_small.pkl',
        'visuals/check-in/displacement-sornot2/recognizer_train_info_sornot_displacement.pkl'
        ]
    labels = ['Noise', 'Rotation', 'Displacement (small)', 'Displacement (large)']
    
    points = []

    for fname in filenames:

        info = pickle.load(open(fname, "rb"))

        y = info['val_accuracies']
        x = [i+1 for i in range(len(y))]

        x = x[:crop_to_first]
        y = y[:crop_to_first]
        points.append((x, y))
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plots = []
    for i, (x, y) in enumerate(points):
        my_max = max(y)
        this_lab = labels[i]
        print(f"Max for {this_lab}: {my_max}")
        plots.append(plt.plot(x, y))
    plt.legend(labels)
    plt.title("Validation Accuracy of Game 2 by Transformation")
    plt.xlabel("Examples Shown (10,000s)")
    plt.ylabel("Accuracy %")
    plt.show()


def vis_omniglot(dataset_src='new-omni'):
    TO_SHOW = 1

    dataset = CustomOmniglot(dataset_src)
    loader = DataLoader(dataset, batch_size=TO_SHOW, shuffle=True)

    # Just look at one batch
    for (X, y) in loader:
        for i in range(TO_SHOW):
            img1 = X[i][0]
            img2 = X[i][1]

            plt.imshow(img1.detach().cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(img2.detach().cpu().permute(1, 2, 0))
            plt.show()

            print(y[i])
        break

    SHOW_LRS = false

    train_info_file = "omniglot-tests/func_omni_info.pkl"
    info = pickle.load(open(train_info_file, "rb"))

    y = info['val_accuracies']
    x = [i+1 for i in range(len(y))]

    # only look at first _ epochs
    crop_to_first = 50
    x = x[:crop_to_first]
    y = y[:crop_to_first]
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plt.plot(x, y)
    plt.title("Validation Accuracy of Omniglot Game")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.show()


def vis_omniglot_lrmatch(dataset_src='omni-lrmatch'):
    TO_SHOW = 1

    dataset = CustomOmniglot(dataset_src)
    loader = DataLoader(dataset, batch_size=TO_SHOW, shuffle=True)

    # Just look at one batch
    for (X, y) in loader:
        for i in range(TO_SHOW):
            img1 = X[i][0]
            img2 = X[i][1]

            plt.imshow(img1.detach().cpu().permute(1, 2, 0))
            plt.show()

            plt.imshow(img2.detach().cpu().permute(1, 2, 0))
            plt.show()

            print(y[i])
        break

    SHOW_LRS = false

    train_info_file = "visuals/final/omni-lrmatch/func_omni_lrmatch_info.pkl"
    info = pickle.load(open(train_info_file, "rb"))

    y = info['val_accuracies']
    x = [i+1 for i in range(len(y))]

    # only look at first _ epochs
    crop_to_first = 50
    x = x[:crop_to_first]
    y = y[:crop_to_first]
    
    if SHOW_LRS:
        # Place vertical lines where learning rate was changed
        for k in info['lrs']:
            if k >= crop_to_first or k == 0:
                continue
            plt.axvline(x = k, color='r', linestyle='dotted')

    plt.plot(x, y)
    plt.title("Validation Accuracy of Omniglot Game")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.show()


    


if __name__ == '__main__':

    # show_dataset('long_match_or_no')
    # plot_recognizer_training()
    # plot_multiple()
    vis_omniglot()
    # vis_omniglot_lrmatch()
