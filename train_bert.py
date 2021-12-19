import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import BERTCONFIG

from preprocess import MidiDataset
from utils import cosine_scheduler, emb_to_index, fix_random_seeds, requires_grad
from bert import PianoBERT

if __name__ == "__main__":
    config = BERTCONFIG()
    # init
    fix_random_seeds(seed=config.SEED)

    dataset = MidiDataset(seq_len=config.SEQ_LENGTH)
    loader = DataLoader(dataset, shuffle=True, batch_size=config.BATCH_SIZE, drop_last=True)

    learning_rate_scheduler = cosine_scheduler(
        base_value=1e-3,
        final_value=1e-7,
        epochs=config.EPOCHS,
        niter_per_ep=len(loader),
        warmup_epochs=50,
        start_warmup_value=0,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = PianoBERT(
        embedding_dim=config.EMBEDDING_DIM,
        seq_length=config.SEQ_LENGTH,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    ).to(device)
    # model.load_state_dict(torch.load("checkpoints_l_w_norm_MLP/model_epoch-1000.pt")["model_state_dict"])
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    writer = SummaryWriter()
    best_loss = 1000000.0
    for epoch in range(config.EPOCHS):
        running_loss = 0
        for i, (X, Y, mask) in tqdm.tqdm(
            enumerate(loader, 0), total=len(loader), ncols=100, desc="Progress"
        ):
            it = len(loader) * epoch + i
            for _, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = learning_rate_scheduler[it]
            writer.add_scalar("Learning rate", learning_rate_scheduler[it], it)

            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            # mask = torch.stack([mask] * model.embedding_dim, dim=-1).to(device)
            # x_mask = torch.stack([mask] * model.embedding_dim, dim=-1).to(device)
            # y_mask = torch.stack([mask] * 8, dim=-1).to(device)

            optimizer.zero_grad()

            # Output
            outputs = model(X)

            # Loss Compute
            # CE Loss
            losses = 0
            for ii, output in enumerate(outputs):
                _mask = torch.stack([mask] * config.EMBEDDING_DIM_LEN[ii], dim=-1)
                output = output.masked_select(_mask).view(-1, config.EMBEDDING_DIM_LEN[ii])
                target = Y[...,ii].masked_select(mask)
                loss = F.cross_entropy(output, target)
                writer.add_scalar(f"Loss/{config.NOTE_PROPERTIES[ii]}", loss.item(), it)
                losses += loss

            # MSE Loss
            # loss = F.mse_loss(
            #     outputs.masked_select(mask),
            #     model.note_embed(Y).detach().masked_select(mask),
            # )
            running_loss += losses.item()

            # Backward Pass
            losses.backward()
            optimizer.step()

            # Log
            writer.add_scalar("Loss", losses.item(), it)

        print(f"Epoch         : {epoch+1}")
        print(f"Learning Rate : {learning_rate_scheduler[it]:4.9f}")
        print(f"Model Loss    : {running_loss/len(loader):4.9f}")
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if (epoch + 1) % 50 == 0:
            best_loss = running_loss / len(loader)
            torch.save(state_dict, f"checkpoints/model_epoch-{epoch+1}.pt")
        # elif best_loss > (running_loss) / len(loader):
        #     torch.save(state_dict, f"checkpoints/model_epoch-{epoch+1}.pt")
    writer.close()
