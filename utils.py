import torch
import numpy as np


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def emb_to_index(outputs, model):
    B, N, C = outputs.shape
    outputs = outputs.view(-1, model.embedding_dim)
    indices_outputs = []
    embedding_dim = model.embedding_dim // 8

    emb = model.note_embed
    embs = [
        emb.octave_embedding,
        emb.pitch_embedding,
        emb.short_duration_embedding,
        emb.medium_duration_embedding,
        emb.long_duration_embedding,
        emb.velocity_embedding,
        emb.short_shift_embedding,
        emb.long_shift_embedding,
    ]
    # embs_dim = [8, 12, 10, 10, 10, 16, 20, 10] 
    for i, emb in enumerate(embs):
        distances = (
            torch.sum(outputs[..., embedding_dim*i:embedding_dim*i+embedding_dim] ** 2, dim=1, keepdim=True)
            + torch.sum(emb.weight[:-3] ** 2, dim=1)
            - 2 * torch.matmul(outputs[..., embedding_dim*i:embedding_dim*i+embedding_dim], emb.weight[:-3].T)
        )
        print(distances)
        encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
        indices_outputs.append(encoding_indices)
    indices_outputs = torch.concat(indices_outputs, dim=-1)
    indices_outputs = indices_outputs.reshape(B, N, 8)
    return indices_outputs

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def emb_distance(outputs, model):
    outputs = outputs.view(-1, model.embedding_dim)
    embedding_dim = model.embedding_dim // 8

    embs = [
        model.note_embed.octave_embedding,
        model.note_embed.pitch_embedding,
        model.note_embed.short_duration_embedding,
        model.note_embed.medium_duration_embedding,
        model.note_embed.long_duration_embedding,
        model.note_embed.velocity_embedding,
        model.note_embed.short_shift_embedding,
        model.note_embed.long_shift_embedding,
    ]
    print(outputs.shape)
    distances = []
    for i, emb in enumerate(embs):
        distance = (
            torch.sum(outputs[..., embedding_dim*i:embedding_dim*i+embedding_dim] ** 2, dim=1, keepdim=True)
            + torch.sum(emb.weight[:-3] ** 2, dim=1)
            - 2 * torch.matmul(outputs[..., embedding_dim*i:embedding_dim*i+embedding_dim], emb.weight[:-3].T)
        )
        distances.append(distance)
        print(distance.shape)
    return distances

def gradient_clipping(model, clip=2.0):
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm()
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
