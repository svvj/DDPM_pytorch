import os
import torch
import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.optim import Adam

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import show_images
from Simple_UNet import SimpleUnet

from main import load_transformed_dataset

IMG_SIZE = 64
BATCH_SIZE = 64


# model: SimpleUnet
# trained parameters: model.pth
# training data: StanfordCars
# training script: main.py

# Load the test dataset and save the test images
datasets = load_transformed_dataset()       # return torch.utils.data.ConcatDataset([train, test])
test_dataset = datasets.datasets[1]         # test dataset
show_images(test_dataset, num_samples=20)   # show the first 20 images

# Load the model
model = SimpleUnet()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# Load the first image from the test dataset
output_images = []
test_img_num = len(test_dataset)
# use tqdm to show the progress bar
for i in tqdm.tqdm(range(20)):
    img = test_dataset[i][0].unsqueeze(0)
    with torch.no_grad():
        output = model(img, torch.tensor([0.5]))  # timesteps = 0.5
    output_images.append(output)

# Show the first 20 images
show_images(output_images, num_samples=20)
plt.show()

# Save the images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(20):
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = (img + 1) / 2
    img = img.clamp(0, 1)
    img = img.cpu().detach().numpy()
    plt.imsave(os.path.join(output_dir, f"output_{i}.png"), img)
print("Images saved successfully!")

