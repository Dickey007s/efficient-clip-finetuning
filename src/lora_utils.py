# -*- coding: utf-8 -*-
"""
lora_utils.py
LoRA 注入工具：为 open_clip ViT 的 attention Q/V 投影注入低秩适配器。

背景：
  torch.nn.MultiheadAttention 把 Q/K/V 融合为单个 in_proj_weight (Parameter，
  shape [3*dim, dim])，不是 nn.Linear 模块，因此 peft 的 target_modules 无法
  匹配，导致 QKV 投影被完全漏掉。标准 LoRA-CLIP 做法 (Hintor/LoRA-CLIP) 对
  Q、V 投影注入 LoRA。

方案：
  用自定义 AttentionWithLoRA 替换每个 resblock 的 attn；
  只对 Q/V projection 注入 LoRA；
  K、out_proj 和 MLP 保持冻结。
  所有原始权重从原 MHA 拷贝过来并冻结，只训练 LoRA 分支。这样保证：
    1) 注入前数学等价于原 CLIP（LoRA 初值 B=0，增量为 0）
    2) Q/V 能被 LoRA 适配，参数量约 147K (rank=4, 12层)
    3) 与论文中 LoRA 微调 attention 的标准做法一致
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    冻结的原始 Linear + 可训练的低秩 LoRA 分支。
    forward: y = W x + (alpha/r) * B A x
    """
    def __init__(self, original: nn.Linear, rank=4, alpha=8):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False
        self.scaling = alpha / rank
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B 初始化为 0 -> 初始增量为 0，注入瞬间数学等价

    def forward(self, x):
        out = self.original(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out + delta


class AttentionWithLoRA(nn.Module):
    """
    手动实现的多头注意力，Q/V 挂 LoRA（标准 LoRA-CLIP 做法），
    K 和 out_proj 保持冻结。行为等价于原 MultiheadAttention。

    open_clip 的 ViT 使用 batch_first=True，输入/输出均为 [N, L, D]。
    forward 调用约定与 nn.MultiheadAttention 一致，返回 (output, None)，
    以兼容 ResidualAttentionBlock.attention 中的 self.attn(...)[0]。
    """
    def __init__(self, mha: nn.MultiheadAttention, rank=4, alpha=8):
        super().__init__()
        self.num_heads = mha.num_heads
        self.embed_dim = mha.embed_dim
        self.head_dim = mha.head_dim
        self.batch_first = getattr(mha, "batch_first", False)
        device = mha.in_proj_weight.device

        # 从融合的 in_proj_weight 拆出 q/k/v
        W = mha.in_proj_weight  # [3d, d]
        b = mha.in_proj_bias    # [3d] or None
        d = self.embed_dim

        def make_linear(Wseg, bseg, trainable=False):
            lin = nn.Linear(d, d, bias=bseg is not None).to(device)
            with torch.no_grad():
                lin.weight.copy_(Wseg)
                if bseg is not None:
                    lin.bias.copy_(bseg)
            for p in lin.parameters():
                p.requires_grad = trainable
            return lin

        # Q 和 V 挂 LoRA（可训练），K 冻结
        q_lin = make_linear(W[:d], b[:d] if b is not None else None, trainable=True)
        k_lin = make_linear(W[d:2*d], b[d:2*d] if b is not None else None, trainable=False)
        v_lin = make_linear(W[2*d:], b[2*d:] if b is not None else None, trainable=True)

        # out_proj 冻结（不挂 LoRA）
        out_lin = nn.Linear(self.embed_dim, self.embed_dim,
                            bias=mha.out_proj.bias is not None).to(device)
        with torch.no_grad():
            out_lin.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                out_lin.bias.copy_(mha.out_proj.bias)
        for p in out_lin.parameters():
            p.requires_grad = False

        # 只给 Q 和 V 挂 LoRA
        self.q_proj = LoRALinear(q_lin, rank=rank, alpha=alpha)
        self.k_proj = k_lin  # 裸 Linear，冻结
        self.v_proj = LoRALinear(v_lin, rank=rank, alpha=alpha)
        self.out_proj = out_lin  # 裸 Linear，冻结

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        """
        与 nn.MultiheadAttention 等价，自动适配 batch_first 设置。
        返回: (output, None)  兼容 [0] 取值
        """
        # 统一转置为 [N, L, D] 处理
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        N, L, D = query.shape
        h = self.num_heads
        dh = self.head_dim

        q = self.q_proj(query).reshape(N, L, h, dh).transpose(1, 2)  # [N, h, L, dh]
        k = self.k_proj(key).reshape(N, L, h, dh).transpose(1, 2)
        v = self.v_proj(value).reshape(N, L, h, dh).transpose(1, 2)

        scale = 1.0 / math.sqrt(dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [N, h, L, L]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [N, h, L, dh]
        out = out.transpose(1, 2).reshape(N, L, D)  # [N, L, D]
        out = self.out_proj(out)

        # 还原为原始格式
        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, None


def inject_lora_into_visual(visual_model, rank=4, alpha=8):
    """
    在 CLIP visual (VisionTransformer) 上注入轻量 LoRA（只适配 Q/V attention）：
      - attn: 替换为 AttentionWithLoRA（Q/V 有 LoRA，K/out 冻结）
      - mlp: 保持原样，不注入 LoRA
    所有原始权重保持冻结，只训练 LoRA 分支。
    返回 LoRA 参数列表。
    """
    blocks = visual_model.transformer.resblocks
    lora_params = []

    for block in blocks:
        # 替换 attention（只 Q/V 有 LoRA）
        old_attn = block.attn
        new_attn = AttentionWithLoRA(old_attn, rank=rank, alpha=alpha)
        block.attn = new_attn

    # 冻结所有非 LoRA 参数
    for p in visual_model.parameters():
        p.requires_grad = False
    # 重新打开 LoRA 参数
    for name, p in visual_model.named_parameters():
        if 'lora_' in name:
            p.requires_grad = True
            lora_params.append(p)

    return lora_params
