import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Converts an image to MNIST format (28x28 grayscale)."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match MNIST
    img = np.array(img) / 255.0  # Normalize to [0,1]
    
    # Show the processed image before prediction
    plt.imshow(img, cmap="gray")
    plt.title("Processed Image")
    # plt.show()
    
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch & channel dim
    return img


if __name__ == "__main__":
    print("Preprocessing script is ready.")

