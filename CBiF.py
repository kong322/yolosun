import torch
import torch.nn as nn
import torch.nn.functional as F

class TopkRouting(nn.Module):
    def __init__(self, dim, num_heads=8, topk=4):
        super().__init__()
        self.topk = topk
        self.num_heads = num_heads
        self.qkv_linear = nn.Linear(dim, dim * 3)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        qkv = self.qkv_linear(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, N, d
        # 简化路由：计算相似度，取topk
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        topk_weights, topk_indices = attn.topk(self.topk, dim=-1)
        # 用topk权重加权v
        v_selected = torch.gather(v.unsqueeze(2).expand(-1, -1, q.size(2), -1, -1), 
                                   dim=3, 
                                   index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, v.size(-1)))
        weighted_v = (topk_weights.softmax(dim=-1).unsqueeze(-1) * v_selected).sum(dim=3)
        out = (weighted_v * q).sum(dim=2)  # 简化，实际更复杂
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

class CBiF(nn.Module):
    """
    简化版双层路由注意力模块，适用于即插即用。
    参数：
        dim: 输入通道数
        num_heads: 注意力头数（默认8）
        topk: 选择的最相关区域数（默认4）
    """
    def __init__(self, dim, num_heads=8, topk=4):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.routing = TopkRouting(dim, num_heads, topk)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.routing(x)
        x = self.proj(x)
        return x + identity