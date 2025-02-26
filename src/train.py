import os
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from src.dataset import Text2ImageDataset
from src.model import Generator, Discriminator
from src.utils import transform, weights_init, save_checkpoint
from datasets import load_dataset
import time
import logging

log_dir = "outputs/log/"
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),  
    filemode="a", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_message(message):
    print(message) 
    logging.info(message) 



def train():
    # Load config
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = config["batch_size"]
    LR = config["learning_rate"]
    EPOCHS = config["epochs"]
    DATASET = config["dataset"]
    L1_COEFF = config["l1_coeff"]
    L2_COEFF = config["l2_coeff"]
    # Load dataset
    ds = load_dataset(DATASET)
    dataset = Text2ImageDataset(ds,transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    embed_dim = 768
    noise_dim = 100
    embed_out_dim = 64
    real_label = 1.
    fake_label = 0.

    generator = Generator(channels=3, embed_dim=embed_dim, 
                          noise_dim=noise_dim, embed_out_dim=embed_out_dim).to(DEVICE)
    generator.apply(weights_init)

    discriminator = Discriminator(channels=3, embed_dim=embed_dim,
                                   embed_out_dim=embed_out_dim).to(DEVICE)
    discriminator.apply(weights_init)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    for epoch in range(EPOCHS):
        epoch_D_loss = []
        epoch_G_loss = []
        batch_time = time.time()

        for idx, batch in enumerate(dataloader):

            images = batch['right_images'].to(DEVICE)
            wrong_images = batch['wrong_images'].to(DEVICE)
            embeddings = batch['right_embed'].to(DEVICE)
            batch_size = images.size(0)


            # Clear gradients for the discriminator
            optimizer_D.zero_grad()

            # Generate random noise
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=DEVICE)

            # Generate fake image batch with the generator
            fake_images = generator(noise, embeddings)

            # Forward pass real batch and calculate loss
            real_out, real_act = discriminator(images, embeddings)
            d_loss_real = criterion(real_out, torch.full_like(real_out, real_label, device=DEVICE))

            # Forward pass wrong batch and calculate loss
            wrong_out, wrong_act = discriminator(wrong_images, embeddings)
            d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, fake_label, device=DEVICE))

            # Forward pass fake batch and calculate loss
            fake_out, fake_act = discriminator(fake_images.detach(), embeddings)
            d_loss_fake = criterion(fake_out, torch.full_like(fake_out, fake_label, device=DEVICE))

            # Compute total discriminator loss
            d_loss = d_loss_real + d_loss_wrong + d_loss_fake

            # Backpropagate the gradients
            d_loss.backward()

            # Update the discriminator
            optimizer_D.step()


            # Clear gradients for the generator
            optimizer_G.zero_grad()

            # Generate new random noise
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=DEVICE)

            # Generate new fake images using Generator
            fake_images = generator(noise, embeddings)

            # Get discriminator output for the new fake images
            out_fake, act_fake = discriminator(fake_images, embeddings)

            # Get discriminator output for the real images
            out_real, act_real = discriminator(images, embeddings)

            # Calculate losses
            g_bce = criterion(out_fake, torch.full_like(out_fake, real_label, device=DEVICE))
            g_l1 = L1_COEFF * l1_loss(fake_images, images)
            g_l2 = L2_COEFF * l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())

            # Compute total generator loss
            g_loss = g_bce + g_l1 + g_l2

            # Backpropagate the gradients
            g_loss.backward()

            # Update the generator
            optimizer_G.step()

            epoch_D_loss.append(d_loss.item())
            epoch_G_loss.append(g_loss.item())

        log_message("Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}".format(
            epoch+1, idx+1, len(dataloader),
            d_loss.mean().item(),
            g_loss.mean().item(),
            time.time() - batch_time
        ))
    t = time.time()
    save_checkpoint(generator.state_dict(),f'models/generator_{t}.pth')
    save_checkpoint(discriminator.state_dict(),f'models/discriminator_{t}.pth')
    log_message("Training complete!!!!")

if __name__ == "__main__":
    train()
