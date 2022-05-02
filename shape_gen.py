import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
from scipy import ndimage, misc
import pickle
import os
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import rotate
import numpy as np

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

device = "cpu"

def get_blank_image():
    return torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(device)

def get_random_color():
    return torch.tensor(np.random.rand(3)).to(device)

# tensor where each location has a value equal to its x coordinate in the tensor
itensor = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(device)
for i in range(3):
    for k in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            itensor[i, j, k] = j  # x coord
# tensor where each location has a value equal to its y coordinate in the tensor
ytensor = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(device)
for i in range(3):
    for k in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            ytensor[i, j, k] = k  # y coord

class Rectangle():
    def __init__(self, height=None, width=None, color=None, x=None, y=None, rotation=None):
        # For height and width - take random 0-1, make "percent of image covered"
        if height is None:
            randpercent = random.random()
            height = int(randpercent * IMAGE_HEIGHT)//2
        if width is None:
            randpercent = random.random()
            width = int(randpercent * IMAGE_WIDTH)//2
        if color is None:
            color = get_random_color()

        self.height = height
        self.width = width
        self.color = color
        self.rotation = rotation

        if x is None:
            x = self.get_random_x()
        if y is None:
            y = self.get_random_y()

        self.x = x
        self.y = y
    
    def get_random_x(self):
        return random.randint(0, IMAGE_WIDTH-self.width)

    def get_random_y(self):
        return random.randint(0, IMAGE_HEIGHT-self.height)

    # X and y are top left on rectangle
    def get_image(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        blank = get_blank_image()
        clrs = blank[:, y:y+self.height, x:x+self.width]
        clrs[0] = self.color[0]
        clrs[1] = self.color[1]
        clrs[2] = self.color[2]

        if self.rotation is not None:
            blank = rotate(blank, self.rotation)
        return blank

class Oval():
    def __init__(self, a=None, b=None, color=None, x=None, y=None, rotation=None):
        
        # For height and width - take random 0-1, make "percent of image covered"
        if b is None:
            randpercent = random.random()
            b = int(randpercent * IMAGE_HEIGHT) // 4
        if a is None:
            randpercent = random.random()
            a = int(randpercent * IMAGE_WIDTH) // 4
            if a <= 0:
                a = 1
        if color is None:
            color = get_random_color()

        self.a = a
        self.b = b

        self.height = 2 * b
        self.width = 2 * a
        self.color = color
        self.rotation = rotation

        if x is None:
            x = self.get_random_x()
        if y is None:
            y = self.get_random_y()

        self.x = x
        self.y = y
    
    def get_random_x(self):
        return random.randint(self.a, IMAGE_WIDTH-self.a)

    def get_random_y(self):
        return random.randint(self.b, IMAGE_WIDTH-self.b)
    
    # X and y are top left on rectangle
    def get_image(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        blank = get_blank_image()

        # IDEA: use fast matrix operations to calculate equations for each coordinate
        # Then use condition matrix in a .where
        mytensor = self.b**2 * (1 - (itensor - x)**2 / self.a**2)
        mytensor = mytensor**.5
        mytensor = (mytensor >= ytensor - y) * (-mytensor <= ytensor - y)

        colormatrix = torch.zeros((3, IMAGE_HEIGHT, IMAGE_WIDTH)).to(device)
        colormatrix[0, :, :] = self.color[0]
        colormatrix[1, :, :] = self.color[1]
        colormatrix[2, :, :] = self.color[2]

        blank = torch.where(mytensor, colormatrix, blank)

        if self.rotation is not None:
            blank = rotate(blank, self.rotation)
        return blank

class Line(Rectangle):
    def __init__(self, length=None, width=None, angle=None, color=None, x=None, y=None):
        if angle is None:
            angle = random.randrange(0, 90)
        if width is None:
            width = round(random.random() * IMAGE_WIDTH / 1.5)
        Rectangle.__init__(self, height=IMAGE_HEIGHT//20, width=width, color=color, x=x, y=y, rotation=angle)

def make_image(k=4):

    # ALGORITHM:
    # 1. Randomly select <= k shapes (with repeats)
    # 2. Place those shapes on the canvas

    blank = get_blank_image()
    shapes = [Rectangle, Oval, Line]
    
    for _ in range(k):
        i = random.randrange(0, len(shapes))
        my_shape = shapes[i]()
        shape_img = my_shape.get_image()
        blank = torch.where(shape_img>.04, shape_img, blank)

    return blank

def make_image_meta(k=4):
    blank = get_blank_image()
    shapes = [Rectangle, Oval, Line]

    indy_shapes = []
    
    for _ in range(k):
        i = random.randrange(0, len(shapes))
        my_shape = shapes[i]()
        shape_img = my_shape.get_image()
        indy_shapes.append(my_shape)
        blank = torch.where(shape_img>.04, shape_img, blank)

    return blank, indy_shapes

def img_perturb_rotate(img):

    # transform = T.Compose([T.GaussianBlur(kernel_size=3),
    #                        T.RandomRotation(degrees=15)])
    transform = T.Compose([T.RandomRotation(degrees=360)])
    img = transform(img)

    return img

def img_perturb_sandp(img):

    upper_threshold = .98
    lower_threshold = .02

    upper_value = 1
    lower_value = 0

    # Create random matrix - values in [0, 1)
    # All entries above value have value put to max
    # All entries below value have value put to min
    random_mat = torch.rand(img.shape)

    new_img = torch.clone(img)

    new_img[random_mat >= upper_threshold] = upper_value
    new_img[random_mat <= lower_threshold] = lower_value

    return new_img


def left_right_match_displace(N=10_000, directory='generated', base_filename='examples_', speak_every=500, to_show=0):

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Gives map of index to file path
    header_obj = {
        'paths': {},  # index -> path
        'desc': '3 pictures are shown. Third is either a repeat of the first or second. Player determines which.',
        'len': 0  # Size of dataset
    }

    for i in range(N):
        imgs = [0, 0, 0]  # placeholder 0s
        imgs[0] = make_image_meta()
        imgs[1] = make_image_meta()
        
        chosen_index = int(random.random() < .5)
        chosen = imgs[chosen_index]

        blank = get_blank_image()
        displacements_x = torch.normal(mean=torch.tensor([0. for _ in range(len(imgs[0][1]))]), std=torch.tensor([5. for _ in range(len(imgs[0][1]))]))
        displacements_y = torch.normal(mean=torch.tensor([0. for _ in range(len(imgs[0][1]))]), std=torch.tensor([5. for _ in range(len(imgs[0][1]))]))
        for k, shape in enumerate(chosen[1]):
            shape.x = round(max(min(displacements_x[k].item()+shape.x, IMAGE_WIDTH - shape.width), 0))
            shape.y = round(max(min(displacements_y[k].item()+shape.y, IMAGE_HEIGHT - shape.height), 0))
            shape_img = shape.get_image()
            blank = torch.where(shape_img>.04, shape_img, blank)
        imgs[2] = blank

        if to_show > 0:
            plt.imshow(imgs[0][0].cpu().permute(1, 2, 0))
            plt.show()
            plt.imshow(imgs[1][0].cpu().permute(1, 2, 0))
            plt.show()
            plt.imshow(imgs[2].cpu().permute(1, 2, 0))
            plt.show()
            to_show -= 1

        obj = {'img1': imgs[0][0], 'img2': imgs[1][0], 'img3': imgs[2], 'y': torch.tensor([int(chosen_index == 1), int(chosen_index == 0)])}

        fname = base_filename + str(i) + ".pkl"
        fname = os.path.join(directory, fname)
        with open(fname, "wb") as fhand:
            pickle.dump(obj, fhand)

        header_obj['paths'][i] = fname
        header_obj['len'] += 1
        if i % speak_every == 0:
            print(f"Written file {i}/{N}")

    fname = "header.pkl"
    fname = os.path.join(directory, fname)
    with open(fname, "wb") as fhand:
        pickle.dump(header_obj, fhand)


def left_right_match(N=10_000, directory='generated', base_filename='examples_', speak_every=500, perturb_func=img_perturb_sandp, to_show=0):

    """
    # Number of examples to write out per file
    k = 250

    # Number of files to write out
    N = 100

    # Format:
    # [{'img1': np.array, 'img2':np.array, 'img3':np.array, 'y': [int first, int second]}]

    directory = "generated"
    base_filename = "examples_"

    speak_every = 10"""

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Gives map of index to file path
    header_obj = {
        'paths': {},  # index -> path
        'desc': '3 pictures are shown. Third is either a repeat of the first or second. Player determines which.',
        'len': 0  # Size of dataset
    }

    for i in range(N):
        img1 = make_image()
        img2 = make_image()
        
        use_first = random.random() < .5
        if use_first:
            img3 = perturb_func(img1)
        else:
            img3 = perturb_func(img2)

        if to_show > 0:
            plt.imshow(img1.cpu().permute(1, 2, 0))
            plt.show()
            plt.imshow(img2.cpu().permute(1, 2, 0))
            plt.show()
            plt.imshow(img3.cpu().permute(1, 2, 0))
            plt.show()
            to_show -= 1

        obj = {'img1': img1, 'img2': img2, 'img3': img3, 'y': torch.tensor([int(use_first), int(not use_first)])}

        fname = base_filename + str(i) + ".pkl"
        fname = os.path.join(directory, fname)
        with open(fname, "wb") as fhand:
            pickle.dump(obj, fhand)

        header_obj['paths'][i] = fname
        header_obj['len'] += 1
    
        if i % speak_every == 0:
            print(f"Written file {i}/{N}")

    fname = "header.pkl"
    fname = os.path.join(directory, fname)
    with open(fname, "wb") as fhand:
        pickle.dump(header_obj, fhand)


def long_match_or_no(N=10_000, n=3, directory='long_match_or_no', base_filename='examples_', speak_every=500, perturb_func=img_perturb_sandp, to_show=0):

    """
    # Number of examples to write out per file
    k = 250

    # Number of files to write out
    N = 100

    # Format:
    # [{'img1': np.array, 'img2':np.array, 'img3':np.array, 'y': [int first, int second]}]

    directory = "generated"
    base_filename = "examples_"

    speak_every = 10"""

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Gives map of index to file path
    header_obj = {
        'paths': {},  # index -> path
        'num_pics': n,
        'desc': 'Player is shown n images. Last one is either repeat or not. Player determines which.',
        'len': 0  # Size of dataset
    }

    for i in range(N):
        images = []
        for _ in range(n-1):
            images.append(make_image())
        
        repeated = random.randrange(n-1)
        is_repeat = random.random() < .5
        last_image = perturb_func(images[repeated]) if is_repeat else perturb_func(make_image())
        images.append(last_image)

        # 1 0 is repeat; 0 1 no repeat
        obj = {'images': images, 'y': torch.tensor([int(is_repeat), int(not is_repeat)])}

        fname = base_filename + str(i) + ".pkl"
        fname = os.path.join(directory, fname)
        with open(fname, "wb") as fhand:
            pickle.dump(obj, fhand)

        header_obj['paths'][i] = fname
        header_obj['len'] += 1

        if to_show > 0:
            for k in range(n):
                plt.imshow(images[k].cpu().permute(1, 2, 0))
                plt.show()
            print(f"Is repeat: {is_repeat}")
            print(obj['y'])
            to_show -= 1
    
        if i % speak_every == 0:
            print(f"Written file {i}/{N}")

    fname = "header.pkl"
    fname = os.path.join(directory, fname)
    with open(fname, "wb") as fhand:
        pickle.dump(header_obj, fhand)
    

def long_match_or_no_displace(N=10_000, n=3, directory='long_match_or_no', base_filename='examples_', speak_every=500, perturb_func=img_perturb_sandp, to_show=0):

    """
    # Number of examples to write out per file
    k = 250

    # Number of files to write out
    N = 100

    # Format:
    # [{'img1': np.array, 'img2':np.array, 'img3':np.array, 'y': [int first, int second]}]

    directory = "generated"
    base_filename = "examples_"

    speak_every = 10"""

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Gives map of index to file path
    header_obj = {
        'paths': {},  # index -> path
        'num_pics': n,
        'desc': 'Player is shown n images. Last one is either repeat or not. Player determines which.',
        'len': 0  # Size of dataset
    }

    for i in range(N):
        images = []
        for _ in range(n-1):
            images.append(make_image_meta())
        
        repeated = random.randrange(n-1)
        is_repeat = random.random() < .5

        blank = get_blank_image()
        if is_repeat:
            displacements_x = torch.normal(mean=torch.tensor([0. for _ in range(len(images[0][1]))]), std=torch.tensor([5. for _ in range(len(images[0][1]))]))
            displacements_y = torch.normal(mean=torch.tensor([0. for _ in range(len(images[0][1]))]), std=torch.tensor([5. for _ in range(len(images[0][1]))]))
            shapes = []
            for k, shape in enumerate(images[repeated][1]):
                shape.x = round(max(min(displacements_x[k].item()+shape.x, IMAGE_WIDTH - shape.width), 0))
                shape.y = round(max(min(displacements_y[k].item()+shape.y, IMAGE_HEIGHT - shape.height), 0))
                shape_img = shape.get_image()
                blank = torch.where(shape_img>.04, shape_img, blank)
                shapes.append(shape)
            images.append((blank, shapes))
        else:
            images.append(make_image_meta())

        # 1 0 is repeat; 0 1 no repeat
        obj = {'images': [x[0] for x in images], 'y': torch.tensor([int(is_repeat), int(not is_repeat)])}

        fname = base_filename + str(i) + ".pkl"
        fname = os.path.join(directory, fname)
        with open(fname, "wb") as fhand:
            pickle.dump(obj, fhand)

        header_obj['paths'][i] = fname
        header_obj['len'] += 1

        if to_show > 0:
            for k in range(n):
                plt.imshow(images[k][0].cpu().permute(1, 2, 0))
                plt.show()
            print(f"Is repeat: {is_repeat}")
            print(obj['y'])
            to_show -= 1
    
        if i % speak_every == 0:
            print(f"Written file {i}/{N}")

    fname = "header.pkl"
    fname = os.path.join(directory, fname)
    with open(fname, "wb") as fhand:
        pickle.dump(header_obj, fhand)


if __name__ == '__main__':
    # long_match_or_no(perturb_func=img_perturb_rotate)
    long_match_or_no_displace()