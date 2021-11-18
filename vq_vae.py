import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
    
    def forward(self, x):

        # flatten
        flat_x = x.view(-1, self.embedding_dim)

        # calculate distance
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                    +torch.sum(self.embedding.weight**2,dim=1)
                    -2*torch.matmul(flat_x, self.embedding.weight.T))
        
        # encoding
        encoding_indices = torch.argmax(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view()

        return x