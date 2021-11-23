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


# def emb_to_index(outputs, model):
#     B, N, C = outputs.shape
#     outputs = outputs.view(-1, 64)
#     quantized_outputs = []
#     emb = model.encoder.note_embed
#     embs = [
#         emb.octave_embedding,
#         emb.pitch_embedding,
#         emb.short_duration_embedding,
#         emb.medium_duration_embedding,
#         emb.long_duration_embedding,
#         emb.velocity_embedding,
#         emb.short_shift_embedding,
#         emb.long_shift_embedding,
#     ]
#     embs_dim = [8, 12, 10, 10, 10, 16, 20, 10]
#     for i, emb in enumerate(embs):
#         distances = (
#             torch.sum(outputs[..., 8*i:8*i+8] ** 2, dim=1, keepdim=true)
#             + torch.sum(emb.weight ** 2, dim=1)
#             - 2 * torch.matmul(outputs[..., 8*i:8*i+8], emb.weight.t)
#         )
#         encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], embs_dim[i], device=outputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
#         quantized = torch.matmul(encodings, emb.weight[:-1]).view(B, N, 8)
#         quantized_outputs.append(quantized)
#     quantized_outputs = torch.concat(quantized_outputs, dim=-1)
#     return quantized_outputs


def emb_to_index(outputs, model):
    B, N, C = outputs.shape
    outputs = outputs.view(-1, 32)
    indices_outputs = []

    emb = model.encoder.note_embed
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
            torch.sum(outputs[..., 4*i:4*i+4] ** 2, dim=1, keepdim=True)
            + torch.sum(emb.weight[:-1] ** 2, dim=1)
            - 2 * torch.matmul(outputs[..., 4*i:4*i+4], emb.weight[:-1].T)
        )
        encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
        indices_outputs.append(encoding_indices)
    indices_outputs = torch.concat(indices_outputs, dim=-1)
    indices_outputs = indices_outputs.reshape(B, N, 8)
    return indices_outputs

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag