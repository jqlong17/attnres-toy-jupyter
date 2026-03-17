from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽量让多次实验结果保持可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)


def generate_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一个非常轻量的合成序列任务。

    这里不使用真实文本，而是构造带有简单规律的整数序列：
    - 每个样本先随机选一个起点 base
    - 再随机选一个步长 step
    - 再叠加一点小噪声 noise

    最终任务是 next-token prediction：
    用前 seq_len 个 token 预测后 seq_len 个 token。

    这样设计的好处是：
    - 训练非常快，适合个人电脑
    - 模型确实需要学习序列规律
    - 足够用来观察残差连接机制，而不是追求真实语言建模效果
    """

    base = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    step = torch.randint(1, 4, (batch_size, 1), device=device)
    noise = torch.randint(0, 3, (batch_size, seq_len + 1), device=device)
    positions = torch.arange(seq_len + 1, device=device).unsqueeze(0)
    seq = (base + step * positions + noise) % vocab_size
    return seq[:, :-1], seq[:, 1:]


@dataclass
class Config:
    """小型 Transformer 的核心超参数。

    参数说明：
    - vocab_size: token 种类数
    - seq_len: 输入序列长度
    - d_model: token 隐藏向量维度
    - n_heads: 注意力头数
    - d_ff: 前馈网络中间层宽度
    - n_layers: Transformer 层数
    - dropout: dropout 比例；这里默认设为 0，便于观察机制本身
    """

    vocab_size: int = 32
    seq_len: int = 24
    d_model: int = 96
    n_heads: int = 4
    d_ff: int = 192
    n_layers: int = 6
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """最基础的因果自注意力模块。

    这里的 "因果" 意味着当前位置只能看见自己和前面的 token，
    不能偷看未来 token，因此适合 next-token prediction。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入形状: [batch, seq_len, d_model]。"""

        batch, seq_len, dim = x.shape

        # 一次线性层同时得到 query、key、value，随后再拆开。
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            # 变成 [batch, heads, seq_len, head_dim]，便于多头并行计算。
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # 标准缩放点积注意力。
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 上三角 mask 把未来位置挡住，确保模型不能看到未来 token。
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        probs = attn.softmax(dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(out)


class FeedForward(nn.Module):
    """Transformer 中注意力后的前馈网络。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthAttentionResidual(nn.Module):
    """Attention Residuals 的核心模块。

    标准残差做的是：
        新状态 = 旧状态 + 当前层输出

    而这里做的是：
        先收集所有历史深度状态 states
        再通过一个沿“深度方向”的注意力权重，把它们加权混合

    论文里提到每一层有一个 learned pseudo-query，这里就是 self.query。
    它不依赖序列位置，而是用来决定“当前层更想看历史上的哪几层”。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 把历史状态堆叠成 [layers_so_far, batch, seq_len, d_model]。
        stacked = torch.stack(states, dim=0)

        # 先做归一化，让不同深度状态在相近尺度下比较。
        normalized = self.norm(stacked)

        # 用一个可学习 query 与每层状态做打分，得到深度方向的 logits。
        # 输出形状 [layers_so_far, batch, seq_len]。
        logits = torch.einsum("d,lbsd->lbs", self.query, normalized)

        # 对深度维做 softmax，得到每一层对历史状态的选择权重。
        weights = logits.softmax(dim=0)

        # 再把这些权重作用回历史状态，得到混合后的输入表示。
        mixed = torch.einsum("lbs,lbsd->bsd", weights, stacked)
        if return_weights:
            return mixed, weights.detach()
        return mixed, None


class UniformAverageResidual(nn.Module):
    """一个简单消融：不学习深度权重，而是对历史层做均匀平均。

    这个变体的作用是回答一个关键问题：
    Attention Residuals 的收益，到底来自“看所有历史层”，
    还是来自“有选择地、非均匀地看所有历史层”？
    """

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        stacked = torch.stack(states, dim=0)
        mixed = stacked.mean(dim=0)
        if return_weights:
            num_states = stacked.size(0)
            batch, seq_len = stacked.shape[1], stacked.shape[2]
            weights = torch.full(
                (num_states, batch, seq_len),
                1.0 / num_states,
                device=stacked.device,
            )
            return mixed, weights.detach()
        return mixed, None


class CrossDepthAttentionResidual(nn.Module):
    """更强的深度方向 cross-attention。

    和论文里的静态 pseudo-query 不同，这里让“当前最新状态”自己生成 query，
    再去和所有历史深度状态做匹配。这样每个 token 都能按输入内容动态决定：
    当前更该从哪几层读取信息。

    这个方向和 DeepCrossAttention 更接近，也更符合“attention is all you need”
    的思想：让选择历史层的方式也完全由注意力来决定。
    """

    def __init__(self, d_model: int, depth_dim: int | None = None) -> None:
        super().__init__()
        depth_dim = depth_dim or max(16, d_model // 4)
        self.query_proj = nn.Linear(d_model, depth_dim, bias=False)
        self.key_proj = nn.Linear(d_model, depth_dim, bias=False)
        self.latest_gate = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        stacked = torch.stack(states, dim=0)
        normalized = self.norm(stacked)

        latest = normalized[-1]
        query = self.query_proj(latest)
        keys = self.key_proj(normalized)

        logits = torch.einsum("bsd,lbsd->lbs", query, keys) / math.sqrt(query.size(-1))
        weights = logits.softmax(dim=0)
        mixed = torch.einsum("lbs,lbsd->bsd", weights, stacked)

        # 给最新状态保留一条可学习直通路径，避免过度平滑。
        gate = torch.sigmoid(self.latest_gate)
        output = gate * states[-1] + (1.0 - gate) * mixed
        if return_weights:
            return output, weights.detach()
        return output, None


class BaselineBlock(nn.Module):
    """标准 Transformer block，对照组使用。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 记录进入这一层前的平均范数，用于观察深度方向的幅值变化。
        input_norm = x.norm(dim=-1).mean().detach()

        # 标准 PreNorm 写法：先归一化，再经过子层，然后做残差相加。
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        metrics = {
            "hidden_norm": input_norm,
            # output_norm 可用于观察这一层处理后幅值有没有继续放大。
            "output_norm": x.norm(dim=-1).mean().detach(),
        }
        return x, metrics


class AttnResBlock(nn.Module):
    """带 Attention Residuals 的 Transformer block。"""

    def __init__(self, config: Config) -> None:
        super().__init__()

        # 分别在 attention 前和 FFN 前做一次深度残差聚合。
        self.attn_res = DepthAttentionResidual(config.d_model)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ff_res = DepthAttentionResidual(config.d_model)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 第一步：先从所有历史状态中“选”出适合 attention 子层的输入。
        attn_input, attn_weights = self.attn_res(states, return_weights=return_weights)
        input_norm = attn_input.norm(dim=-1).mean().detach()

        # 和标准 Transformer 一样，子层内部依然有残差。
        attn_out = attn_input + self.attn(self.attn_norm(attn_input))

        # attention 处理后的输出也加入候选历史状态中，供 FFN 前再次聚合。
        states_after_attn = states + [attn_out]
        ff_input, ff_weights = self.ff_res(
            states_after_attn, return_weights=return_weights
        )
        out = ff_input + self.ff(self.ff_norm(ff_input))
        metrics = {
            "hidden_norm": input_norm,
            "output_norm": out.norm(dim=-1).mean().detach(),
        }
        if attn_weights is not None:
            # 对 batch 和 seq_len 取平均，得到每层整体偏好的深度权重。
            metrics["attn_depth_weights"] = attn_weights.mean(dim=(1, 2)).cpu()
        if ff_weights is not None:
            metrics["ff_depth_weights"] = ff_weights.mean(dim=(1, 2)).cpu()
        return out, metrics


class MeanResBlock(nn.Module):
    """均匀深度聚合版本，用来和 AttnRes 做消融比较。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_res = UniformAverageResidual()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ff_res = UniformAverageResidual()
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_input, attn_weights = self.attn_res(states, return_weights=return_weights)
        input_norm = attn_input.norm(dim=-1).mean().detach()
        attn_out = attn_input + self.attn(self.attn_norm(attn_input))
        states_after_attn = states + [attn_out]
        ff_input, ff_weights = self.ff_res(
            states_after_attn, return_weights=return_weights
        )
        out = ff_input + self.ff(self.ff_norm(ff_input))
        metrics = {
            "hidden_norm": input_norm,
            "output_norm": out.norm(dim=-1).mean().detach(),
        }
        if attn_weights is not None:
            metrics["attn_depth_weights"] = attn_weights.mean(dim=(1, 2)).cpu()
        if ff_weights is not None:
            metrics["ff_depth_weights"] = ff_weights.mean(dim=(1, 2)).cpu()
        return out, metrics


class LayerScaleBlock(nn.Module):
    """类似 ReZero/LayerScale 的残差门控变体。

    它仍然只使用当前层输出，但不再固定用系数 1 相加，
    而是学习两个可训练缩放参数，分别控制 attention 和 FFN 子层的残差强度。
    这是文献里另一类很常见的残差创新方向。
    """

    def __init__(self, config: Config, init_scale: float = 0.1) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.attn_scale = nn.Parameter(torch.full((config.d_model,), init_scale))
        self.ff_scale = nn.Parameter(torch.full((config.d_model,), init_scale))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_norm = x.norm(dim=-1).mean().detach()
        x = x + self.attn_scale * self.attn(self.attn_norm(x))
        x = x + self.ff_scale * self.ff(self.ff_norm(x))
        metrics = {
            "hidden_norm": input_norm,
            "output_norm": x.norm(dim=-1).mean().detach(),
        }
        return x, metrics


class DepthCrossBlock(nn.Module):
    """使用 content-dependent depth cross-attention 的 block。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_res = CrossDepthAttentionResidual(config.d_model)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ff_res = CrossDepthAttentionResidual(config.d_model)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_input, attn_weights = self.attn_res(states, return_weights=return_weights)
        input_norm = attn_input.norm(dim=-1).mean().detach()
        attn_out = attn_input + self.attn(self.attn_norm(attn_input))
        states_after_attn = states + [attn_out]
        ff_input, ff_weights = self.ff_res(
            states_after_attn, return_weights=return_weights
        )
        out = ff_input + self.ff(self.ff_norm(ff_input))
        metrics = {
            "hidden_norm": input_norm,
            "output_norm": out.norm(dim=-1).mean().detach(),
        }
        if attn_weights is not None:
            metrics["attn_depth_weights"] = attn_weights.mean(dim=(1, 2)).cpu()
        if ff_weights is not None:
            metrics["ff_depth_weights"] = ff_weights.mean(dim=(1, 2)).cpu()
        return out, metrics


class DepthCrossLiteBlock(nn.Module):
    """更轻量的版本：每层只在 attention 前做一次深度 cross-attention。

    这样可以明显减少计算量，同时保留“按输入动态选择历史层”的核心思想。
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_res = CrossDepthAttentionResidual(config.d_model)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        states: list[torch.Tensor],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_input, attn_weights = self.attn_res(states, return_weights=return_weights)
        input_norm = attn_input.norm(dim=-1).mean().detach()
        attn_out = attn_input + self.attn(self.attn_norm(attn_input))
        out = attn_out + self.ff(self.ff_norm(attn_out))
        metrics = {
            "hidden_norm": input_norm,
            "output_norm": out.norm(dim=-1).mean().detach(),
        }
        if attn_weights is not None:
            metrics["attn_depth_weights"] = attn_weights.mean(dim=(1, 2)).cpu()
        return out, metrics


class TinyTransformerLM(nn.Module):
    """最小可运行语言模型骨架。

    这里只保留最关键的结构：
    token embedding + position embedding + 多层 block + lm head。

    variant:
    - baseline: 使用标准残差
    - attnres: 使用深度注意力残差
    - meanres: 使用均匀深度平均，作为 AttnRes 的消融对照
    - layerscale: 使用可学习残差缩放，代表另一类残差创新
    - depthcross: 用当前状态作为 query，对历史深度状态做 cross-attention
    - depthcross_lite: 轻量 depthcross，每层只做一次深度 cross-attention
    """

    def __init__(self, config: Config, variant: str = "baseline") -> None:
        super().__init__()
        if variant not in {
            "baseline",
            "attnres",
            "meanres",
            "layerscale",
            "depthcross",
            "depthcross_lite",
        }:
            raise ValueError(
                "variant must be 'baseline', 'attnres', 'meanres', 'layerscale', 'depthcross', or 'depthcross_lite'"
            )
        self.config = config
        self.variant = variant
        self.depth_repeats = 1
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [
                BaselineBlock(config)
                if variant == "baseline"
                else MeanResBlock(config)
                if variant == "meanres"
                else LayerScaleBlock(config)
                if variant == "layerscale"
                else DepthCrossBlock(config)
                if variant == "depthcross"
                else DepthCrossLiteBlock(config)
                if variant == "depthcross_lite"
                else AttnResBlock(config)
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]] | None]:
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)

        # token embedding 和位置编码相加，构成 Transformer 的输入。
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]

        metrics: list[dict[str, torch.Tensor]] = []
        if self.variant == "baseline":
            for _ in range(self.depth_repeats):
                for block in self.blocks:
                    x, block_metrics = block(x)
                    metrics.append(block_metrics)
        elif self.variant in {"attnres", "meanres", "depthcross", "depthcross_lite"}:
            states = [x]
            for _ in range(self.depth_repeats):
                for block in self.blocks:
                    x, block_metrics = block(states, return_weights=return_metrics)
                    states.append(x)
                    metrics.append(block_metrics)
        else:
            for _ in range(self.depth_repeats):
                for block in self.blocks:
                    x, block_metrics = block(x)
                    metrics.append(block_metrics)

        # 最后映射到词表维度，得到每个位置对下一个 token 的 logits。
        logits = self.lm_head(self.final_norm(x))
        if return_metrics:
            return logits, metrics
        return logits, None


def clone_model(model: nn.Module) -> nn.Module:
    """深拷贝模型，保留相同参数但互不影响。"""

    return copy.deepcopy(model)


def run_training(
    model: TinyTransformerLM,
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cpu",
) -> dict[str, list[float] | list[list[float]]]:
    """训练一个 toy 模型，并记录可视化所需的历史信息。"""

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: dict[str, list] = {
        "loss": [],
        "layer_norms": [],
    }
    for step in range(steps):
        # 每一步都动态生成一批新样本，避免过拟合固定小数据。
        x, y = generate_batch(
            batch_size, model.config.seq_len, model.config.vocab_size, device
        )

        # return_metrics=True 方便同时收集每层范数和深度权重。
        logits, metrics = model(x, return_metrics=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # 标准反向传播流程。
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["layer_norms"].append(
            [float(item["hidden_norm"]) for item in metrics or []]
        )

        if (
            model.variant in {"attnres", "meanres", "depthcross", "depthcross_lite"}
            and metrics
        ):
            # 保存每一步中每一层的深度注意力权重，便于后续分析。
            history.setdefault("attn_depth_weights", []).append(
                [
                    item.get("attn_depth_weights", torch.tensor([])).tolist()
                    for item in metrics
                ]
            )

        if (step + 1) % 50 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")
    return history


@torch.no_grad()
def collect_depth_statistics(
    model: TinyTransformerLM,
    batches: int = 8,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """在评估模式下收集不同深度的统计量。

    这里不会更新参数，只是多采样几批数据，计算：
    - 每层输入隐藏状态的平均范数
    - AttnRes 在深度维上的平均权重
    """

    model.to(device)
    model.eval()
    norms = []
    depth_weights = []
    for _ in range(batches):
        x, _ = generate_batch(
            batch_size, model.config.seq_len, model.config.vocab_size, device
        )
        _, metrics = model(x, return_metrics=True)
        norms.append(
            torch.tensor([float(item["hidden_norm"]) for item in metrics or []])
        )
        if (
            model.variant in {"attnres", "meanres", "depthcross", "depthcross_lite"}
            and metrics
        ):
            rows = []
            for item in metrics:
                weights = item.get("attn_depth_weights")
                if weights is not None:
                    rows.append(weights)
            if rows:
                # 不同层能看到的历史状态数不同，因此先补齐长度再求平均。
                max_len = max(row.numel() for row in rows)
                padded = []
                for row in rows:
                    if row.numel() < max_len:
                        row = F.pad(row, (0, max_len - row.numel()))
                    padded.append(row)
                depth_weights.append(torch.stack(padded))
    stats = {"hidden_norms": torch.stack(norms).mean(0)}
    if depth_weights:
        stats["attn_depth_weights"] = torch.stack(depth_weights).mean(0)
    return stats


@torch.no_grad()
def evaluate_loss(
    model: TinyTransformerLM,
    batches: int = 12,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """用多批随机数据估计训练后模型的平均 loss。"""

    model.to(device)
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = generate_batch(
            batch_size, model.config.seq_len, model.config.vocab_size, device
        )
        logits, _ = model(x, return_metrics=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss))
    return sum(losses) / len(losses)


def compare_variants(
    variants: list[str],
    config: Config,
    seeds: list[int],
    steps: int = 120,
    batch_size: int = 32,
    lr: float = 3e-4,
    eval_batches: int = 12,
    device: str = "cpu",
) -> dict[str, dict[str, float | list[float]]]:
    """对多个残差变体做同配置比较，输出每个变体的均值和波动。"""

    results: dict[str, dict[str, float | list[float]]] = {}
    for variant in variants:
        final_losses = []
        eval_losses = []
        for seed in seeds:
            set_seed(seed)
            model = TinyTransformerLM(config, variant=variant)
            history = run_training(
                model,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            final_losses.append(history["loss"][-1])
            eval_losses.append(
                evaluate_loss(
                    model,
                    batches=eval_batches,
                    batch_size=batch_size,
                    device=device,
                )
            )

        mean_final = sum(final_losses) / len(final_losses)
        mean_eval = sum(eval_losses) / len(eval_losses)
        std_eval = (
            sum((loss - mean_eval) ** 2 for loss in eval_losses) / len(eval_losses)
        ) ** 0.5
        results[variant] = {
            "final_train_losses": final_losses,
            "eval_losses": eval_losses,
            "mean_final_train_loss": mean_final,
            "mean_eval_loss": mean_eval,
            "std_eval_loss": std_eval,
        }

    return results


def _finalize_plot(save_path: str | Path | None = None) -> None:
    """统一处理图像保存和展示。"""

    plt.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: str | Path | None = None,
) -> None:
    """绘制训练 loss 曲线。"""

    plt.figure(figsize=(8, 4))
    for name, history in histories.items():
        plt.plot(history["loss"], label=name)
    plt.xlabel("training step")
    plt.ylabel("cross entropy loss")
    plt.title("Toy next-token training")
    plt.legend()
    plt.grid(alpha=0.3)
    _finalize_plot(save_path)


def plot_hidden_norms(
    stats: dict[str, dict[str, torch.Tensor]],
    save_path: str | Path | None = None,
) -> None:
    """绘制各层隐藏状态范数，观察是否随深度明显增长。"""

    plt.figure(figsize=(8, 4))
    for name, item in stats.items():
        values = item["hidden_norms"].cpu().numpy()
        plt.plot(range(1, len(values) + 1), values, marker="o", label=name)
    plt.xlabel("layer")
    plt.ylabel("mean hidden-state norm")
    plt.title("Hidden-state magnitude across depth")
    plt.legend()
    plt.grid(alpha=0.3)
    _finalize_plot(save_path)


def plot_attnres_heatmap(
    stats: dict[str, torch.Tensor],
    save_path: str | Path | None = None,
) -> None:
    """绘制 AttnRes 在深度维上的选择热力图。"""

    weights = stats.get("attn_depth_weights")
    if weights is None:
        raise ValueError("No attn depth weights found")
    plt.figure(figsize=(7, 4))
    plt.imshow(weights.cpu().numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="average depth attention weight")
    plt.xlabel("source depth state")
    plt.ylabel("transformer layer")
    plt.title("AttnRes depth selection")
    _finalize_plot(save_path)
