import os
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning as L

# Define a simple feedforward neural network for classification
class SimpleClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes (MNIST digits)

    def forward(self, x):
        # Forward pass through the network
        x = x.view(x.size(0), -1)  # Flatten the image from 28x28 to a vector of size 28*28
        x = nn.ReLU()(self.fc1(x))  # ReLU activation after first fully connected layer
        x = self.fc2(x)  # Output layer (logits for each class)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # Forward pass
        loss = nn.CrossEntropyLoss()(logits, y)  # Cross entropy loss for classification
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # Forward pass
        test_loss = nn.CrossEntropyLoss()(logits, y)  # Cross entropy loss
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        return optimizer

# Data transformations and loaders
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=7)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

# Define the trainer
trainer = L.Trainer(limit_test_batches=1.0, max_epochs=100, log_every_n_steps=5)

# Initialize the classifier model
classifier = SimpleClassifier()

# Train and Test
if __name__ == "__main__":
    # Train the model
    trainer.fit(model=classifier, train_dataloaders=train_loader)

    # Test the model
    trainer.test(model=classifier, dataloaders=test_loader)

    # Save the trained model as a PyTorch file
    torch.save(classifier.state_dict(), 'mnist_classifier.pth')
    print("Model saved as mnist_classifier.pth.")

    # Convert to ONNX
    classifier.eval()  # Set to evaluation mode

    # Define a dummy input for ONNX export
    dummy_input = torch.randn(1, 28 * 28)  # Batch size of 1, flattened MNIST image

    # Export to ONNX
    onnx_file_path = 'mnist_classifier.onnx'
    torch.onnx.export(
        classifier,                  # Model to export
        dummy_input,                 # Input tensor
        onnx_file_path,              # Output file path
        export_params=True,          # Store the trained parameters
        opset_version=13,            # ONNX opset version
        do_constant_folding=True,    # Optimize constant folding for inference
        input_names=['input'],       # Model's input name
        output_names=['output'],     # Model's output name
        dynamic_axes={               # Dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_file_path}.")
