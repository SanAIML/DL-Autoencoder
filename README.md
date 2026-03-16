# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The problem addressed by this experiment is image denoising. The objective is to develop and train a Denoising Autoencoder using PyTorch to effectively remove synthetic Gaussian noise from MNIST handwritten digit images, thereby reconstructing the original, clean images.

## DESIGN STEPS
### STEP 1: 


Set up the PyTorch environment, define image transformations, load the MNIST dataset, create data loaders, and define the add_noise function to simulate noisy inputs for training the autoencoder.
### STEP 2: 
Construct the DenoisingAutoencoder model using convolutional layers for the encoder and transposed convolutional layers for the decoder to learn an efficient representation and reconstruct the images.



### STEP 3: 
Instantiate the autoencoder model, move it to the computational device (CPU/GPU), define the Mean Squared Error (MSE) loss function, and configure the Adam optimizer for model training.


### STEP 4: 

Implement and execute the training loop for a specified number of epochs, where the model learns to reconstruct the original images from their noisy counterparts by minimizing the defined loss function.

### STEP 5: 

Develop a function to evaluate the trained model's performance by visualizing a set of original, noisy, and the corresponding denoised images from the test set to qualitatively assess the autoencoder's denoising capability.







## PROGRAM

### Name:Sanchita Sandeep

### Register Number: 212224240142

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
      running_loss = 0.0
      for images, _ in loader:
        images = images.to(device)
        noisy_images = add_noise(images).to(device)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

      print(f"Epoch[{epoch+1}/{epochs}],Loss: {running_loss/len(loader):.4f}")
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:      Eesha Ranka             ")
    print("Register Number:        212224240040          ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



```

### OUTPUT

### Model Summary


### Training loss

## Original vs Noisy Vs Reconstructed Image
Include a few sample images here.

## RESULT
The model is succesfully created
