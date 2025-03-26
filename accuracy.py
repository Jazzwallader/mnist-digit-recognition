import torch
from model import MNISTModel  # Import the trained model
from dataset import get_data_loaders  # Import test dataset

# Load the test dataset
_, test_loader = get_data_loaders(batch_size=64)  # Load only test data

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Compute Accuracy
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get highest probability prediction
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Print Accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
