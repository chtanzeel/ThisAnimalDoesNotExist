import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import torch.nn as nn

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps * 8, feature_maps * 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps * 4, feature_maps * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps * 2, feature_maps, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps, img_channels, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
latent_dim = 128
img_channels = 3
feature_maps = 64

# Load the trained generator
checkpoint_path = "gan_checkpoint_final.pth"  # Ensure this file is uploaded to your Hugging Face Space

# Initialize and load the generator model
netG = Generator(latent_dim, img_channels, feature_maps).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
netG.load_state_dict(checkpoint['netG_state_dict'])
netG.eval()  # Set model to evaluation mode

print("✅ Model loaded successfully!")


# Function to denormalize images
def denormalize(img_tensor):
    return img_tensor * 0.5 + 0.5  # Convert from [-1,1] to [0,1]


# Function to generate images
def generate_images():
    batch_size = 8  # Number of images per batch
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        generated_images = netG(noise)

    # Denormalize images
    generated_images_denorm = denormalize(generated_images)

    # Create an image grid
    img_grid = torchvision.utils.make_grid(generated_images_denorm, nrow=8, padding=2, normalize=True)

    # Convert tensor to NumPy
    np_img = img_grid.cpu().numpy().transpose((1, 2, 0))  # (C, H, W) → (H, W, C)

    # Save image temporarily
    plt.figure(figsize=(12, 12))
    plt.imshow(np.clip(np_img, 0, 1))
    plt.axis("off")
    plt.savefig("generated.png", bbox_inches='tight')
    plt.close()

    return "generated.png"


# Create Gradio Interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[],
    outputs=gr.Image(type="filepath"),
    title="✨ GAN Animal Faces Generator ✨",
    description="Click the button below to generate a new batch of AI-generated animal faces!",
    allow_flagging="never"
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
