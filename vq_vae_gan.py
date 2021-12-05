import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MLP
from preprocess import MidiDataset
from swin_encoder import EncoderBlock, SwinEncoder
from utils import emb_to_index


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_loss=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss = commitment_loss

        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, max_norm=1, norm_type=2
        )
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
    def __init__(
        self,
        embed_dim,
        num_layers,
        num_heads,
        window_size,
        attn_drop,
        mlp_drop,
        act_layer,
        norm_layer,
    ):
        super().__init__()
        self.high_up = nn.Upsample(scale_factor=4, mode="linear")
        self.mid_up = nn.Upsample(scale_factor=4, mode="linear")
        self.low_up = nn.Upsample(scale_factor=4, mode="linear")
        self.norm_layer1 = nn.LayerNorm([16, embed_dim * 4])
        self.norm_layer2 = nn.LayerNorm([64, embed_dim * 2])
        self.norm_layer3 = nn.LayerNorm([256, embed_dim])
        self.norm_layer4 = nn.LayerNorm([1024, embed_dim])
        self.high_attention = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim * 8,
                    window_size=window_size,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=mlp_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers[0])
            ]
        )
        self.mid_attention = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim * 4,
                    window_size=window_size,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=mlp_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers[1])
            ]
        )
        self.low_attention = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim * 2,
                    window_size=4,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=mlp_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers[2])
            ]
        )
        self.note_attention = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim,
                    window_size=4,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=mlp_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers[3])
            ]
        )
        self.high_mlp = MLP([embed_dim * 8, embed_dim * 4], act_layer, mlp_drop)
        self.mid_mlp = MLP([embed_dim * 4, embed_dim * 2], act_layer, mlp_drop)
        self.low_mlp = MLP([embed_dim * 2, embed_dim], act_layer, mlp_drop)

    def forward(self, high, mid, low):
        """
        high : Tensor
            shape of B x 16 x 8C
        mid : Tensor
            shape of B x 64 x 4C
        low : Tensor
            shape of B x 256 x 2C
        """
        for layer in self.high_attention:
            high = layer(high)
        high = self.norm_layer1(self.high_mlp(high))
        high = self.high_up(high.transpose(-1, -2)).transpose(-1, -2)
        mid = mid + high
        for layer in self.mid_attention:
            mid = layer(mid)
        mid = self.norm_layer2(self.mid_mlp(mid))
        mid = self.mid_up(mid.transpose(-1, -2)).transpose(-1, -2)
        low = low + mid
        for layer in self.low_attention:
            low = layer(low)
        low = self.norm_layer3(self.low_mlp(low))
        low = self.low_up(low.transpose(-1, -2)).transpose(-1, -2)
        for layer in self.note_attention:
            low = layer(low)
        return self.norm_layer4(low)


class Generator(nn.Module):
    def __init__(
        self,
        embed_dim,
        window_size=16,
        num_heads=4,
        num_layers=[4, 4, 4, 4],
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
        self.decoder = Decoder(
            embed_dim,
            num_layers,
            num_heads,
            window_size,
            attn_drop,
            mlp_drop,
            act_layer,
            norm_layer,
        )
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
        high_loss, high_quantized, high_perplexity, _ = self.high_vq(
            F.normalize(high_encoded, dim=-1)
        )
        mid_loss, mid_quantized, mid_perplexity, _ = self.mid_vq(
            F.normalize(mid_encoded, dim=-1)
        )
        low_loss, low_quantized, low_perplexity, _ = self.low_vq(
            F.normalize(low_encoded, dim=-1)
        )

        # Decoder
        decoded = self.decoder(high_quantized, mid_quantized, low_quantized)

        # Loss
        vq_loss = high_loss + mid_loss + low_loss
        x = self.encoder.note_embed(x)
        recon_loss = F.mse_loss(decoded, x)

        decoded_quantized = emb_to_index(decoded, self)
        # recon_loss = torch.nn.CrossEntropyLoss()(
        #     x.view(-1, C).float(), decoded_quantized.view(-1, C).float()
        # )
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
    midi = torch.unsqueeze(midi, 0)
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
    )
    discriminator = Discriminator(embed_dim=16)
    generator.load_state_dict(
        torch.load("checkpoints/model_epoch-40.pt")["gen_state_dict"]
    )
    print(generator.high_vq.embedding.weight)
    encoded = generator.encoder(midi)
    for ii, e in enumerate(encoded):
        encoded[ii] = F.normalize(e, dim=-1)
    fig, ax = plt.subplots(5)
    sns.heatmap(encoded[0][0].detach().numpy(), ax=ax[0])
    sns.heatmap(encoded[1][0].detach().numpy(), ax=ax[1])
    sns.heatmap(encoded[2][0].detach().numpy(), ax=ax[2])
    sns.heatmap(encoded[3][0].detach().numpy(), ax=ax[3])
    sns.heatmap(encoded[4][0].detach().numpy(), ax=ax[4])
    _, high_quantized, high_perplexity, _ = generator.high_vq(encoded[-1])
    _, mid_quantized, mid_perplexity, _ = generator.mid_vq(encoded[-2])
    _, low_quantized, low_perplexity, _ = generator.low_vq(encoded[-3])
    # print(high_perplexity)
    # print(mid_perplexity)
    # print(low_perplexity)
    fig, ax = plt.subplots(3)
    sns.heatmap(high_quantized[0].detach().numpy(), ax=ax[0])
    sns.heatmap(mid_quantized[0].detach().numpy(), ax=ax[1])
    sns.heatmap(low_quantized[0].detach().numpy(), ax=ax[2])
    decoded = generator.decoder(high_quantized, mid_quantized, low_quantized)
    decoded_quantized = emb_to_index(decoded, generator)
    fig, ax = plt.subplots(2)
    sns.heatmap(decoded[0].detach().numpy(), ax=ax[0])
    sns.heatmap(decoded_quantized[0].detach().numpy(), ax=ax[1])
    # print(decoded_quantized.shape)
    # real_fake_value = discriminator(decoded_quantized)
    # print(real_fake_value)

    # print(output.shape)
    plt.show()
