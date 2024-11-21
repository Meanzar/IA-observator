import os
import torch
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning as L

# Define encoder and decoder
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

# Define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = nn.Linear(3, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        reconstruction_loss = nn.functional.mse_loss(x_hat, x)
        
        # Classification
        logits = self.classifier(z)
        classification_loss = nn.functional.cross_entropy(logits, y)
        
        # Combine losses
        loss = reconstruction_loss + classification_loss
        self.log("train_loss", loss)
        self.log("classification_loss", classification_loss)
        self.log("reconstruction_loss", reconstruction_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        logits = self.classifier(z)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
        threshold = 0.5 
        x_binary = (x > threshold).float()
        x_hat_binary = (x_hat > threshold).float()
        accuracy = (x_binary == x_hat_binary).float().mean()
        self.log("reconstruction_accuracy", accuracy)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        self.log("test_accuracy", accuracy)

    def forward(self, x):
        # Define the forward pass
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        return optimizer

# Initialize the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

# Data transformations and loaders
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=7)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

# Define the trainer
trainer = L.Trainer(limit_test_batches=1.0, max_epochs=10, log_every_n_steps=2)

# Train and Test
if __name__ == "__main__":
    # Train the model
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # Test the model
    trainer.test(model=autoencoder, dataloaders=test_loader)

    # Save the trained model as a PyTorch file
    torch.save(autoencoder.state_dict(), 'mnist_autoencoder.pth')
    print("Model saved as mnist_autoencoder.pth.")

  # Convert to ONNX
    # Set the model to evaluation mode
    autoencoder.eval()
    
    # Define a dummy input that matches the model's input shape
    dummy_input = torch.randn(1, 28 * 28)

    # Export the model to ONNX
    onnx_file_path = 'mnist_autoencoder.onnx'
    torch.onnx.export(
        autoencoder,                  # Model to export
        dummy_input,                  # Input tensor
        onnx_file_path,               # Output file path
        export_params=True,           # Store the trained parameter weights inside the model file
        opset_version=13,             # ONNX opset version
        do_constant_folding=True,     # Optimize constant folding for inference
        input_names=['input'],        # Name of the model's input
        output_names=['output'],      # Name of the model's output
        dynamic_axes={                # Support dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_file_path}.")
