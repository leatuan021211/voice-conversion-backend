import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Cross-attention layer for combining two sets of features.
    """
    
    def __init__(self, embed_size):
        """
        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
        """
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, target, source, mask=None):
        Q = self.query(target)
        K = self.key(source)
        V = self.value(source)

        # Perform attention calculation (self or cross)
        out, _ = self.cross_attention(Q, K, V, mask)
        return self.fc_out(out)
        
    def cross_attention(self, Q, K, V, mask=None):
        # Compute the dot products between Q and K, then scale
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to normalize scores and get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights