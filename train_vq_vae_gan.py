import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import MidiDataset
from swin_encoder import SwinEncoder
from utils import cosine_scheduler, emb_to_index, fix_random_seeds, requires_grad
from vq_vae_gan import Discriminator, Generator

if __name__ == "__main__":
    # init
    fix_random_seeds(seed=42)

    # hyperparameters
    epochs = 1000
    batch_size = 20
    critic_frequency = 1
    # log_frequency = 100

    dataset = MidiDataset()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True)


    learning_rate_scheduler = cosine_scheduler(
        base_value=1e-4,
        final_value=1e-6,
        epochs=epochs,
        niter_per_ep=len(loader),
        warmup_epochs=30,
        start_warmup_value=0,
    )
    device = torch.device("cuda")
    # device = torch.device("cpu")

    generator = Generator(
        embed_dim=16,
        window_size=16,
        num_heads=4,
        num_layers=[4, 4, 4, 4],
        downsample_res=4,
        depth=[4, 4, 2, 2],
        codebook_size=[64, 32, 16],
        seq_length=1024,
        attn_drop=0.5,
        proj_drop=0.5,
        mlp_drop=0.5,
        commitment_loss=1.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ).to(device)
    discriminator = Discriminator(
        embed_dim=16,
        window_size=16,
        num_heads=4,
        downsample_res=4,
        depth=[4, 4, 2, 2],
        seq_length=1024,
        attn_drop=0.5,
        proj_drop=0.5,
        mlp_drop=0.5,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ).to(device)
    generator.train()
    discriminator.train()

    gen_optimizer = torch.optim.AdamW(generator.parameters())
    dis_optimizer = torch.optim.AdamW(discriminator.parameters())

    adversarial_loss = nn.BCEWithLogitsLoss()
    valid = (
        torch.Tensor(1, 1024, 1).fill_(0.99).to(device)
        if batch_size == None
        else torch.Tensor(batch_size, 1024, 1).fill_(0.99).to(device)
    )
    fake = (
        torch.Tensor(1, 1024, 1).fill_(0.01).to(device)
        if batch_size == None
        else torch.Tensor(batch_size, 1024, 1).fill_(0.01).to(device)
    )

    writer = SummaryWriter()
    best_loss = 1000000.0
    for epoch in range(epochs):
        running_gen_loss = 0
        running_dis_loss = 0
        for i, midi in tqdm.tqdm(
            enumerate(loader, 0), total=len(loader), ncols=100, desc="progress"
        ):
            it = len(loader) * epoch + i
            for _, param_group in enumerate(gen_optimizer.param_groups):
                param_group["lr"] = learning_rate_scheduler[it]
            for _, param_group in enumerate(dis_optimizer.param_groups):
                param_group["lr"] = learning_rate_scheduler[it]
            writer.add_scalar("Learning rate", learning_rate_scheduler[it], it)

            midi = midi.to(device)

            # Train discriminator
            requires_grad(generator, False)
            requires_grad(discriminator, True)
            dis_optimizer.zero_grad()

            # Generator Output
            outputs, vq_loss, recon_loss, perplexity = generator(midi)

            # Loss Compute
            real_loss = adversarial_loss(discriminator(midi), valid)
            fake_loss = adversarial_loss(discriminator(outputs), fake)
            dis_loss = real_loss + fake_loss
            running_dis_loss += dis_loss.item()

            # Backward Pass
            dis_loss.backward()
            dis_optimizer.step()

            # Log
            writer.add_scalar(
                "Discriminator", dis_loss.item(), it
            )  
            writer.add_scalar(
                "Discriminator/Real Loss", real_loss.item(), it
            )  # Get the real right
            writer.add_scalar(
                "Discriminator/Fake Loss", fake_loss.item(), it
            )  # Get the fake right

            # Train Generator
            if (epoch + 1) % critic_frequency == 0:
                requires_grad(generator, True)
                requires_grad(discriminator, False)
                gen_optimizer.zero_grad()

                # Generator Compute
                outputs, vq_loss, recon_loss, perplexity = generator(midi)

                # Loss Compute (push generated output closer to real)
                realistic_loss = adversarial_loss(discriminator(outputs), valid)
                gen_loss = vq_loss + recon_loss + realistic_loss
                running_gen_loss += gen_loss.item()

                # Backward Pass
                gen_loss.backward()
                gen_optimizer.step()

                log_it = len(loader) * (epoch - critic_frequency + 1) // critic_frequency + i
                writer.add_scalar("Generator/Quantization Loss", vq_loss.item(), log_it)
                writer.add_scalar("Generator/Reconstruction Loss", recon_loss.item(), log_it)
                writer.add_scalar("Generator/Realistic Loss", realistic_loss.item(), log_it)
                writer.add_scalar("Perplexity/High", perplexity[0], log_it)
                writer.add_scalar("Perplexity/Middle", perplexity[1], log_it)
                writer.add_scalar("Perplexity/Low", perplexity[2], log_it)

        print(f"Epoch               : {epoch+1}")
        print(f"Learning Rate       : {learning_rate_scheduler[it]:4.9f}")
        print(f"Discriminator Loss  : {running_dis_loss/len(loader):4.9f}")
        if (epoch + 1) % critic_frequency == 0:
            print(f"Generator Loss      : {running_gen_loss/len(loader):4.9f}")
            running_loss = running_dis_loss + running_gen_loss
            state_dict = {
                "epoch": epoch,
                "gen_state_dict": generator.state_dict(),
                "dis_state_dict": discriminator.state_dict(),
                "gen_optimizer": gen_optimizer.state_dict(),
                "dis_optimizer": dis_optimizer.state_dict(),
            }
            if (epoch + 1) % 20 == 0:
                best_loss = running_loss / len(loader)
                torch.save(state_dict, f"checkpoints/model_epoch-{epoch+1}.pt")
            # elif best_loss > (running_loss) / len(loader):
            #     torch.save(state_dict, f"checkpoints/model_epoch-{epoch+1}.pt")
    writer.close()
