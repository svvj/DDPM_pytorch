import matplotlib.pyplot as plt

def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset. """
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i+1)

        # Access the image from the tuple
        image = img[0]

        # Check if the image has 3 channels and permute dimensions if necessary
        if image.shape[0] == 3:  # From (3, H, W)
            image = image.permute(1, 2, 0)  # To (H, W, 3)

        plt.imshow(image)
