import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from model import MNISTModel
from dataset import get_data_loaders
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from collections import Counter
from dataset import train_labels

# Learning Rate settings
# base_lr = 0.00001
# max_lr = 0.001
# warmup_epochs = 10

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST CNN with custom hyperparameters')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='Base learning rate for the optimizer')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs until reaching max LR')
    parser.add_argument('--epochs', type=int, default=25, help='Number of total training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    args = parser.parse_args()

    # Learning Rate Warmup Function
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs  # Gradually increase LR
        return 1  # After warmup, use the base LR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss func., optimizer and scheduler using argparse values
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Track loss and learning rate
    train_losses = []
    learning_rates = []

    # Load dataset
    train_loader, test_loader = get_data_loaders()
    counter = Counter(train_labels)
    print("Training Data Distribution:", counter)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])  # Track LR changes

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step() # Update learning rate

    # Save model
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved!")

    # Plot Training Loss
    plt.figure(figsize=(10, 4))
    plt.plot(range(4, len(train_losses) + 1), train_losses[3:], marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)


    plt.savefig("training_loss_and_lr.png")  # Save the figure
    plt.show()

if __name__ == '__main__':
    main()