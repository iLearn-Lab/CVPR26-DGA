import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViT(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.backbone = "CLIP-VIT"
        self.patch_embedding = vit_model.conv1
        self.class_embedding = vit_model.class_embedding
        self.positional_embedding = vit_model.positional_embedding
        self.ln_pre = vit_model.ln_pre
        self.blocks = vit_model.transformer.resblocks
        self.ln_post = vit_model.ln_post
        self.proj = vit_model.proj  # not used
        self.out_dim = self.ln_post.bias.shape[0]

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    def forward(self, x):
        x = x.to(self.dtype)
        x = self.patch_embedding(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)
        for i in range(n_layers):
            block = self.blocks[i]

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj_weight = _mlp[0].weight
            _mlp_in_proj_bias = _mlp[0].bias
            _mlp_act = _mlp[1]
            _mlp_out_proj_weight = _mlp[2].weight
            _mlp_out_proj_bias = _mlp[2].bias

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x  # 197 128 768

            x = _ln_1(x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)

            # x = F.scaled_dot_product_attention(q, k, v)
            # scaled_dot_product_attention:
            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)

            x = torch.bmm(attn, v)  # 1536 197 64

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            x = x.view(-1, _bsz, _emb_dim)
            x = x + identity
            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x  # deep copy
            x = _ln_2(x)
            x_out = F.linear(x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            x = x_out
            x = _mlp_act(x)
            x_out = F.linear(x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            x = x_out

            x = x + identity
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        x_cls = x[:, 0, :]
        x_cls = x_cls @ self.proj



        return x_cls

