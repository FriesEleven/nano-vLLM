# 提供张量类型标注和 torch.compile 装饰器。
import torch
# 引入神经网络模块基类。
from torch import nn
# 引入无状态的 SiLU 激活函数。
import torch.nn.functional as F


# 定义 SwiGLU 中融合 SiLU 激活与逐元素乘法的轻量模块。
class SiluAndMul(nn.Module):

    # 编译该热点函数，减少门控 MLP 推理阶段的 Python 调度开销。
    @torch.compile
    # 将合并投影结果均分为 gate/up 两段，并计算 SiLU(gate) * up。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿最后一维将合并的 gate_up 投影均分成两个张量。
        x, y = x.chunk(2, -1)
        # 对 gate 段施加 SiLU 后与 up 段逐元素相乘。
        return F.silu(x) * y
