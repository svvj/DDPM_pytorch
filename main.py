import os
import torch
import matplotlib.pyplot as plt
import kaggle
import tqdm

import torch.nn.functional as F
from torch.optim import Adam

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from utils import show_images
from Simple_UNet import SimpleUnet

IMG_SIZE = 64
BATCH_SIZE = 64


def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into the range [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scales data into the range [-1,1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = datasets.StanfordCars(root="./data", download=False, transform=data_transform)

    test = datasets.StanfordCars(root="./data", download=False, transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scales data into the range [0,1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scales data into the range [0,255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # Converts to numpy
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image.to("cpu")))


def linear_beta_schedule(timesteps, start=0.0001, end=0.2):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cuda"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.rand_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean_variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device=x_0.device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            # show_tensor_image(img.detach().cpu())
    # plt.show()


if __name__ == "__main__":
    # torch device cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    data_path = './data'
    if not os.path.exists(data_path):
        kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path='./data', unzip=True)

    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # show_images(data)
    # plt.show()

    # Define beta schedule
    T = 200
    betas = linear_beta_schedule(timesteps=T)   # (200,)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas     # (200,)
    alphas_cumprod = torch.cumprod(alphas, axis=0) # Returns the cumulative product of elements of input in the dim.
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Pads the input tensor with constant value 1.0
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    #########################################################
    # Simulate forward diffusion
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, (idx//stepsize) + 1)
        image, noise = forward_diffusion_sample(image, t)
        show_tensor_image(image)
    # plt.show()

    #########################################################
    # Import the model
    model = SimpleUnet().to(device)
    print("Num params:", sum(p.numel() for p in model.parameters()))

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100

    for epoch in tqdm.trange(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE, ), device=device).long()
            loss = get_loss(model, batch[0].to(device), t)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item()}")
                sample_plot_image()
