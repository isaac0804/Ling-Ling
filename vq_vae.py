import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.fromnumeric import shape

from encoder import MLP
from preprocess import MidiDataset
from swin_encoder import SwinEncoder
from utils import emb_to_index


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_loss=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss = commitment_loss

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

    def forward(self, inputs):
        input_shape = inputs.shape
        # flatten
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # calculate distance
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embedding.weight.T)
        )

        # encoding
        encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = e_latent_loss + self.commitment_loss * q_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class Decoder(nn.Module):
    def __init__(self, embed_dim, mlp_drop, act_layer):
        super().__init__()
        self.high_up = nn.Upsample(scale_factor=(64, 1), mode="bilinear")
        self.mid_up = nn.Upsample(scale_factor=(16, 1), mode="bilinear")
        self.low_up = nn.Upsample(scale_factor=(4, 1), mode="bilinear")
        self.mlp1 = MLP(
            [embed_dim * 14, embed_dim * 8, embed_dim * 4, embed_dim],
            act_layer,
            mlp_drop,
        )
        self.mlp2 = MLP([embed_dim, embed_dim, embed_dim], act_layer, mlp_drop)
        self.mlp3 = MLP([embed_dim, embed_dim, embed_dim], act_layer, mlp_drop)
        self.mlp4 = MLP([embed_dim, embed_dim, embed_dim], act_layer, mlp_drop)
        self.conv1 = nn.Conv2d(
            1,
            embed_dim // 8,
            (15, embed_dim // 8),
            stride=(1, embed_dim // 8),
            padding=(7, 0),
        )
        self.conv2 = nn.Conv2d(
            1,
            embed_dim // 8,
            (15, embed_dim // 8),
            stride=(1, embed_dim // 8),
            padding=(7, 0),
        )
        self.conv3 = nn.Conv2d(
            1,
            embed_dim // 8,
            (15, embed_dim // 8),
            stride=(1, embed_dim // 8),
            padding=(7, 0),
        )
        self.norm_layer1 = nn.LayerNorm([1024, embed_dim])
        self.norm_layer2 = nn.LayerNorm([1024, embed_dim])
        self.norm_layer3 = nn.LayerNorm([1024, embed_dim])

    def forward(self, high, mid, low):
        """
        high : Tensor
            shape of B x 16 x 512
        mid : Tensor
            shape of B x 64 x 256
        low : Tensor
            shape of B x 256 x 128
        """
        high = torch.stack([high] * 64, dim=1).transpose(1, 2).flatten(1, 2)
        mid = torch.stack([mid] * 16, dim=1).transpose(1, 2).flatten(1, 2)
        low = torch.stack([low] * 4, dim=1).transpose(1, 2).flatten(1, 2)
        x = torch.concat([high, mid, low], dim=-1)
        x = self.mlp1(x)
        if self.train:
            x = torch.randn(x.shape, device=x.device)*(0.1**0.5) + x
        x = self.conv1(torch.unsqueeze(x, 1)).transpose(1, 2).flatten(-2) + x
        x = self.norm_layer1(self.mlp2(x) + x)
        if self.train:
            x = torch.randn(x.shape, device=x.device)*(0.1**0.5) + x
        x = self.conv2(torch.unsqueeze(x, 1)).transpose(1, 2).flatten(-2) + x
        x = self.norm_layer2(self.mlp3(x) + x)
        if self.train:
            x = torch.randn(x.shape, device=x.device)*(0.1**0.5) + x
        x = self.conv3(torch.unsqueeze(x, 1)).transpose(1, 2).flatten(-2) + x
        x = self.norm_layer3(self.mlp4(x) + x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        embed_dim,
        window_size=16,
        num_heads=4,
        downsample_res=4,
        depth=[8, 6, 4, 2],
        codebook_size=[128, 64, 32],
        seq_length=1024,
        attn_drop=0.5,
        proj_drop=0.5,
        mlp_drop=0.5,
        commitment_loss=0.25,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.encoder = SwinEncoder(
            embed_dim,
            window_size,
            num_heads,
            downsample_res,
            depth,
            seq_length,
            attn_drop,
            proj_drop,
            mlp_drop,
            act_layer,
            norm_layer,
        )
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.decoder = Decoder(embed_dim, mlp_drop, act_layer)
        self.low_vq = VectorQuantizer(codebook_size[0], embed_dim * 2, commitment_loss)
        self.mid_vq = VectorQuantizer(codebook_size[1], embed_dim * 4, commitment_loss)
        self.high_vq = VectorQuantizer(codebook_size[2], embed_dim * 8, commitment_loss)

    def forward(self, x):
        # Encoder
        B, N, C = x.shape
        assert N == self.seq_length and C == 8
        encoded = self.encoder(x)
        high_encoded, mid_encoded, low_encoded = encoded[-1], encoded[-2], encoded[-3]

        # Vector Quantize
        high_loss, high_quantized, high_perplexity, _ = self.high_vq(high_encoded)
        mid_loss, mid_quantized, mid_perplexity, _ = self.mid_vq(mid_encoded)
        low_loss, low_quantized, low_perplexity, _ = self.low_vq(low_encoded)

        # Decoder
        decoded = self.decoder(high_quantized, mid_quantized, low_quantized)

        # Loss
        vq_loss = high_loss + mid_loss + low_loss
        decoded_quantized = emb_to_index(decoded, self)
        weights = torch.tensor(
            [0.0625, 0.0625, 0.0938, 0.125, 0.25, 0.0625, 0.0938, 0.25]
        ).to(x.device)
        recon_loss = torch.nn.CrossEntropyLoss(weight=weights)(
            x.view(-1, C).float(), decoded_quantized.view(-1, C).float()
        )
        return (
            decoded_quantized,
            vq_loss,
            recon_loss,
            [high_perplexity, mid_perplexity, low_perplexity],
        )


class Discriminator(nn.Module):
    def __init__(
        self,
        embed_dim,
        window_size=16,
        num_heads=4,
        downsample_res=4,
        depth=[8, 6, 4, 2],
        seq_length=1024,
        attn_drop=0.5,
        proj_drop=0.5,
        mlp_drop=0.5,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.encoder = SwinEncoder(
            embed_dim,
            window_size,
            num_heads,
            downsample_res,
            depth,
            seq_length,
            attn_drop,
            proj_drop,
            mlp_drop,
            act_layer,
            norm_layer,
        )
        self.mlp = MLP(
            [embed_dim * 15, embed_dim * 4, embed_dim, 8, 1],
            act_layer,
            mlp_drop,
        )

    def forward(self, x):
        # Encoder
        enc = self.encoder(x)
        high = torch.stack([enc[-1]] * 64, dim=1).transpose(1, 2).flatten(1, 2)
        mid = torch.stack([enc[-2]] * 16, dim=1).transpose(1, 2).flatten(1, 2)
        low = torch.stack([enc[-3]] * 4, dim=1).transpose(1, 2).flatten(1, 2)
        x = torch.concat([high, mid, low, enc[0]], dim=-1)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    dataset = iter(MidiDataset())
    midi = next(dataset)
    generator = Generator(embed_dim=32)
    discriminator = Discriminator(embed_dim=32)
    generator.load_state_dict(torch.load("checkpoints/model_epoch-30_loss-18.97.pt"))
    encoded = generator.encoder(midi)
    print(midi.shape)
    _, high_quantized, high_perplexity, _ = generator.high_vq(encoded[-1])
    _, mid_quantized, mid_perplexity, _ = generator.mid_vq(encoded[-2])
    _, low_quantized, low_perplexity, _ = generator.low_vq(encoded[-3])
    # print(high_perplexity)
    # print(mid_perplexity)
    # print(low_perplexity)
    # fig, ax = plt.subplots()
    # sns.heatmap(high[0].detach().numpy(), ax=ax)
    # sns.heatmap(mid[0].detach().numpy(), ax=ax)
    # sns.heatmap(low[0].detach().numpy(), ax=ax)
    # plt.show()
    decoded = generator.decoder(high_quantized, mid_quantized, low_quantized)
    print(decoded.shape)
    decoded_quantized = emb_to_index(decoded, generator)
    print(decoded_quantized.shape)
    real_fake_value = discriminator(decoded_quantized)
    print(real_fake_value)

    # output, vq_loss, recon_loss = model(midi)
    # print(output.shape)
