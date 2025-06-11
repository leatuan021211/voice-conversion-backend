import torch
import torch.nn as nn
from torch.nn import functional as F

class VQEmbeddingEMA(nn.Module):

	def __init__(self, n_embeddings:int, embedding_dim:int, epsilon=1e-5):
		super(VQEmbeddingEMA, self).__init__()
		self.epsilon = epsilon

		init_bound = 1 / n_embeddings
		embedding = torch.Tensor(n_embeddings, embedding_dim)
		embedding.uniform_(-init_bound, init_bound)
		embedding = embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-4)
		self.register_buffer("embedding", embedding)
		self.register_buffer("ema_count", torch.zeros(n_embeddings))
		self.register_buffer("ema_weight", self.embedding.clone())

	def instance_norm(self, x, dim, epsilon=1e-5):
		mu = torch.mean(x, dim=dim, keepdim=True)
		std = torch.std(x, dim=dim, keepdim=True)

		z = (x - mu) / (std + epsilon)
		return z


	def forward(self, x: torch.Tensor) -> tuple:

		x = self.instance_norm(x, dim=1)

		embedding = self.embedding / (torch.norm(self.embedding, dim=1, keepdim=True) + 1e-4)

		M, D = embedding.size()
		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(torch.sum(embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, embedding.t(),
                                alpha=-2.0, beta=1.0)

		indices = torch.argmin(distances.float(), dim=-1).detach()
		encodings = F.one_hot(indices, M).float()
		quantized = F.embedding(indices, self.embedding)

		quantized = quantized.view_as(x)

		commitment_loss = F.mse_loss(x, quantized.detach())

		quantized_ = x + (quantized - x).detach()
		quantized_ = (quantized_ + quantized)/2

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		return quantized_, commitment_loss, perplexity