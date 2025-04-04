import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.generate_qkv = nn.Linear(dim, 3 * dim)
        self.dropout = nn.Dropout(attn_drop)
        self.num_heads = num_heads
        self.dim = dim
        self.sqrt_d_k = math.sqrt(dim // num_heads)
        self.output_linear = nn.Linear(dim, dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_toks = x.shape[0], x.shape[1]
        qkv = self.generate_qkv(x)
        qkv = torch.reshape(qkv, (batch_size, num_toks, self.num_heads, -1)).swapaxes(1, 2)
        
        q, k, v = torch.chunk(qkv, 3, -1)

        QK_T = q.matmul(k.transpose(2, 3)) / self.sqrt_d_k
        SQK_T = nn.functional.softmax(QK_T, dim=-1)
        SQK_T = self.dropout(SQK_T)

        result_weights = SQK_T.matmul(v).swapaxes(1, 2).reshape(batch_size, num_toks, self.dim)
    
        return self.output_linear(result_weights)
        

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)

