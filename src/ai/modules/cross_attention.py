import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Cross-attention layer for combining two sets of features.
    """
    
    def __init__(self, query_dim, kv_dim, inner_dim, inner_kv_dim, out_dim, num_heads=8):
        """
        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
        """
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.inner_kv_dim = inner_kv_dim
        self.inner_dim = inner_dim
        # Define linear transformations for Q, K, V
        self.norm_query = nn.LayerNorm(query_dim, elementwise_affine=True)
        self.query = nn.Linear(query_dim, inner_dim)
        self.norm_key = nn.LayerNorm(kv_dim, elementwise_affine=True)
        self.key = nn.Linear(kv_dim, inner_kv_dim)
        self.norm_value = nn.LayerNorm(kv_dim, elementwise_affine=True)
        self.value = nn.Linear(kv_dim, inner_kv_dim)
        
        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(inner_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, target, source, attention_mask=None):
        N, q_seq_len, _ = target.shape
        _, kv_seq_len, dim = source.shape
        Q = self.norm_query(target)
        Q = self.query(Q)
        K = self.norm_key(source)
        K = self.key(K)
        V = self.norm_value(source)
        V = self.value(V)
        
        Q = Q.view(N, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform attention calculation (self or cross)
        out, _ = self.cross_attention(Q, K, V, mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(N, kv_seq_len, self.inner_kv_dim)
        return self.dropout(self.fc_out(out))
        
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