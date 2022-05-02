import pickle
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomImageDataset, LongMatchOrNoGame
import torch
from torch import nn, optim
import statistics

from shape_mem_encoder import MemoryEncoder
from shape_gen import img_perturb_rotate, img_perturb_sandp, long_match_or_no, left_right_match, left_right_match_displace, long_match_or_no_displace

"""

Train functional NN to perform memory task

"""

# Channel, Height, Width
INPUT_SHAPE = (3, 28, 28)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]

MEMORY_SIZE = 128

MEMORY_LENGTH = 2

device = 'cuda'

class FunctionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        # For left or right game
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH,  MEMORY_SIZE * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(MEMORY_SIZE * 2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
            torch.nn.Softmax(dim=1)
        )
        # For seen or not with 2 memories, rotation
        """self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH, (INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH)//4),
            torch.nn.ReLU(),
            torch.nn.LayerNorm((INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH)//4),
            torch.nn.Linear((INPUT_SIZE + MEMORY_SIZE * MEMORY_LENGTH)//4, 5 * MEMORY_SIZE * MEMORY_LENGTH),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(5 * MEMORY_SIZE * MEMORY_LENGTH),
            torch.nn.Linear(5 * MEMORY_SIZE * MEMORY_LENGTH, 5 * MEMORY_SIZE * MEMORY_LENGTH),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(5 * MEMORY_SIZE * MEMORY_LENGTH),
            torch.nn.Linear(5 * MEMORY_SIZE * MEMORY_LENGTH, MEMORY_SIZE * MEMORY_LENGTH),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(MEMORY_SIZE * MEMORY_LENGTH),
            torch.nn.Linear(MEMORY_SIZE * MEMORY_LENGTH, MEMORY_SIZE),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(MEMORY_SIZE),
            torch.nn.Linear(MEMORY_SIZE, 2),
            torch.nn.Softmax(dim=1)
        )"""

    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        return prediction

class MemoryBank():
    def __init__(self) -> None:
        self.rows = MEMORY_LENGTH
        self.cols = MEMORY_SIZE
        self.bank = torch.zeros(self.rows, self.cols, dtype=torch.float).to(device)

    # Moves each row forward one position, placing new memory at 0
    # x should be a tensor
    @torch.no_grad()
    def add(self, x):
        new_mem = [x] + [self.bank[i] for i in range(self.rows-1)]
        self.bank = torch.vstack(new_mem)

if __name__ == '__main__':
    
    recognizer = FunctionalNN().to(device)

    encoder_file = 'encoder-.011.pkl'
    mem_encoder = pickle.load(open(encoder_file, 'rb'))

    learning_rates = {
        0: 1,
        10: .5,
        15: .25,
        25: .1,
        35: .05,
        45: .025,
    }

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(recognizer.parameters(), lr=1)

    num_epochs = 50

    change_data = True  # create new data after every epoch

    game_names = ['left or right noise', 'left or right rotate', 'left or right displacement', 'seen or not noise', 'seen or not rotation', 'seen or not displacement']
    # Note: when adding a game, make sure to change the data creation at the end of the epoch
    game = game_names[3]

    info_file_name = "recognizer_train_info_sornot_noise_try2.pkl"  # output training info name
    model_file_name = "functionalnn_sornot_noise_try2.pkl"  # output model name
    
    memories_to_load = 0
    if game.split()[:-1] == ['seen', 'or', 'not']:
        dataset = LongMatchOrNoGame('long_match_or_no')
        memories_to_load = dataset.num_pics - 1
    elif game.split()[:-1] == ['left', 'or', 'right']:
        dataset = CustomImageDataset('generated')
        memories_to_load = 2
    print(f"Playing {game} with {memories_to_load} memories.")

    train_size = int(.8 * len(dataset))  # .8 for 80%
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    epoch_val_accuracies = []

    for epoch in range(num_epochs):

        # Switch lr according to curve
        if epoch in learning_rates:
            optimizer = torch.optim.SGD(recognizer.parameters(), lr=learning_rates[epoch]) 

        train_loader = DataLoader(train, batch_size=128, shuffle=True)
        val_loader = DataLoader(val, batch_size=128, shuffle=True)

        train_losses = []
        accuracies = []
        val_accuraces = []


        print(f"Epoch #{epoch+1}")

        # Train
        recognizer.train()
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()
            # Select images that are memories
            memories = []
            for i in range(memories_to_load):
                memories.append(X[:, i, :, :, :])
                memories[i] = torch.flatten(memories[i], start_dim=1, end_dim=3)
                memories[i] = mem_encoder.encode(memories[i])

            # Flatten input
            # Note: memories_to_load would equal the index of the first non memory
            sensory_input = X[:, memories_to_load, :, :]
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)
            # Concatenate memories and input
            total_input = torch.cat(memories + [flattened], 1).float()

            pred = recognizer(total_input)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            accuracies.append(batch_acc.item())

        val_losses = []
        val_accuracies = []

        # Validate
        recognizer.eval()
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device).float()

            # Select images that are memories
            memories = []
            for i in range(memories_to_load):
                memories.append(X[:, i, :, :, :])
                memories[i] = torch.flatten(memories[i], start_dim=1, end_dim=3)
                memories[i] = mem_encoder.encode(memories[i])

            # Flatten input
            # Note: memories_to_load would equal the index of the first non memory
            sensory_input = X[:, memories_to_load, :, :]
            flattened = torch.flatten(sensory_input, start_dim=1, end_dim=3)
            # Concatenate memories and input
            total_input = torch.cat(memories + [flattened], 1).float()

            pred = recognizer(total_input)

            loss = loss_fn(pred, y)
            val_losses.append(loss.item())
            
            correctness_tensor = pred.argmax(dim=-1) == y.argmax(dim=-1)
            batch_acc = sum(correctness_tensor)/len(correctness_tensor)
            val_accuracies.append(batch_acc.item())

        current_val_loss = statistics.mean(val_losses)
        val_total_acc = statistics.mean(val_accuracies)
        print(f"Val loss: {current_val_loss}")
        print(f"Val accuracy: {val_total_acc}")
        epoch_val_accuracies.append(val_total_acc)

        if change_data:
            
            if game == 'seen or not noise':
                long_match_or_no(n=dataset.num_pics, perturb_func=img_perturb_sandp)
                dataset = LongMatchOrNoGame('long_match_or_no')
            elif game == 'seen or not rotation':
                long_match_or_no(n=dataset.num_pics, perturb_func=img_perturb_rotate)
                dataset = LongMatchOrNoGame('long_match_or_no')
            elif game == 'seen or not displacement':
                long_match_or_no_displace(n=dataset.num_pics)
                dataset = LongMatchOrNoGame('long_match_or_no')
            elif game == 'left or right displacement':
                left_right_match_displace()
                dataset = CustomImageDataset('generated')
            elif game == 'left or right rotate':
                left_right_match(perturb_func=img_perturb_rotate)
                dataset = CustomImageDataset('generated')
            elif game == 'left or right noise':
                left_right_match(perturb_func=img_perturb_sandp)
                dataset = CustomImageDataset('generated')

            train_size = int(.8 * len(dataset))  # .8 for 80%
            val_size = len(dataset) - train_size
            train, val = random_split(dataset, [train_size, val_size])

    recognizer_train_info = {}
    recognizer_train_info['val_accuracies'] = epoch_val_accuracies
    recognizer_train_info['lrs'] = learning_rates
    recognizer_train_info['num_epochs'] = num_epochs

    pickle.dump(recognizer_train_info, open(info_file_name, 'wb'))
    pickle.dump(recognizer, open(model_file_name, 'wb'))

