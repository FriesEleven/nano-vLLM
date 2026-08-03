# 引入缓存装饰器，避免重复构建相同配置的 RoPE 表。
from functools import lru_cache
# 提供位置频率表和旋转运算所需的张量 API。
import torch
# 引入位置编码模块所继承的基类。
from torch import nn


# 对 Query 或 Key 的每个头向量应用预计算好的旋转位置编码。
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 先转为 float32，再沿头维度均分为每个二维旋转平面的一对分量。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 根据二维旋转矩阵计算每对分量的第一半输出。
    y1 = x1 * cos - x2 * sin
    # 根据二维旋转矩阵计算每对分量的第二半输出。
    y2 = x2 * cos + x1 * sin
    # 拼回完整头维度并转换回输入精度。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# 定义预计算 sin/cos 查找表并对 Query/Key 应用 RoPE 的模块。
class RotaryEmbedding(nn.Module):

    # 按最大位置和旋转基数构建可供索引的 cos/sin 缓存。
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        # 初始化模块基类。
        super().__init__()
        # 保存完整注意力头维度。
        self.head_size = head_size
        # 当前实现要求 RoPE 覆盖整个头维度而非其中一部分。
        assert rotary_dim == head_size
        # 计算每个二维旋转平面的逆频率。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 生成从 0 到最大位置的所有绝对位置下标。
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算每个位置和每个旋转平面的相位角。
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 对相位角求余弦，得到旋转矩阵的对角项。
        cos = freqs.cos()
        # 对相位角求正弦，得到旋转矩阵的非对角项。
        sin = freqs.sin()
        # 将 cos/sin 拼接并插入可广播到注意力头的维度。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 注册非持久缓冲区，使其随设备迁移但不写入 checkpoint。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # 编译位置查表和旋转热点函数，降低每层 RoPE 的运行时开销。
    @torch.compile
    # 根据 token 位置选择对应 sin/cos，并同时旋转 Query 和 Key。
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 从预计算表按每个 token 的位置收集 sin/cos 对。
        cos_sin = self.cos_sin_cache[positions]
        # 将查表结果拆分为旋转计算所需的余弦和正弦项。
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 对 Query 头向量施加位置相关的二维旋转。
        query = apply_rotary_emb(query, cos, sin)
        # 对 Key 头向量施加相同的位置相关二维旋转。
        key = apply_rotary_emb(key, cos, sin)
        # 返回已编码位置信息的 Query 与 Key。
        return query, key


# 只缓存一个最近使用的 RoPE 实例，避免相同模型配置重复分配查找表。
@lru_cache(1)
# 按参数创建或返回缓存的 RotaryEmbedding 模块。
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    # 构造与当前注意力配置匹配的 RoPE 查找表模块。
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    # 交给 lru_cache 保存并返回该模块实例。
    return rotary_emb
