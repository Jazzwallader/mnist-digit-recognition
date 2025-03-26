import torch
from model import MNISTModel
from preprocess import preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="mnist_cnn.pth"):
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

"""def predict_digit(image_path):
    model = load_model()
    img = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(img)
        print("Raw model output: ", output.cpu().numpy())
        _, predicted = torch.max(output, 1)

    print(f"Predicted Digit: {predicted.item()}")
    return predicted.item()
"""

def predict_digit(image_path):
    model = load_model()
    img = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(img, apply_softmax=True)
        print("Raw model output:", output.cpu().numpy())  # Debugging output
        _, predicted = torch.max(output, 1)

    print(f"Predicted Digit: {predicted.item()}")
    return predicted.item()


# Run inference
"""if __name__ == "__main__":
    image_path = "img/captured_digit.png"  # Change this!
    predict_digit(image_path)
"""
