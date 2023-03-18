import torchvision
import matplotlib.pyplot as plt

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)

for i in range(1, 60):
    # Get a sample image
    sample_image, sample_label = train_dataset[i]
        # Display the image
    if sample_label == 7:
        plt.imshow(sample_image, cmap='gray')
        plt.show()
        print(f"Number {sample_label}")