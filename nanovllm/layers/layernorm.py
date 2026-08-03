# 提供张量运算、类型标注和 torch.compile。
import torch
# 引入归一化层继承的模块基类及可训练参数。
from torch import nn


# 定义只按均方根缩放、不减均值的 RMSNorm 层。
class RMSNorm(nn.Module):

    # 创建数值稳定项和每个隐藏通道对应的缩放参数。
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        # 初始化模块基类。
        super().__init__()
        # 保存防止除零的数值稳定常数。
        self.eps = eps
        # 初始化每个隐藏维度的可学习缩放系数为 1。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 编译无残差分支，减少高频归一化调用的 Python 开销。
    @torch.compile
    # 以 float32 计算 RMS 后将结果转换回原始精度并乘以可学习权重。
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 保存输入精度，以便在稳定的 float32 计算后恢复输出精度。
        orig_dtype = x.dtype
        # 将归一化的中间计算提升为 float32，降低半精度误差。
        x = x.float()
        # 计算每个 token 隐藏向量的均方值。
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 原位乘以均方根的倒数，完成 RMS 归一化。
        x.mul_(torch.rsqrt(var + self.eps))
        # 恢复原始精度并施加每通道可学习缩放。
        x = x.to(orig_dtype).mul_(self.weight)
        # 返回无残差输入的归一化结果。
        return x

    # 编译融合残差加法和 RMSNorm 的分支，避免额外读取写入。
    @torch.compile
    # 先累加残差，再返回新的残差值和其 RMSNorm 输出。
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 保存当前子层输出的原始精度。
        orig_dtype = x.dtype
        # 在 float32 中原位相加当前输出和延迟传递的残差。
        x = x.float().add_(residual.float())
        # 将相加后的未归一化值保存为下一子层要使用的新残差。
        residual = x.to(orig_dtype)
        # 计算融合残差后向量的均方值。
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 原位执行均方根倒数缩放。
        x.mul_(torch.rsqrt(var + self.eps))
        # 恢复原始精度并应用可学习的通道缩放。
        x = x.to(orig_dtype).mul_(self.weight)
        # 同时返回下一子层输入和继续延迟传递的残差。
        return x, residual

    # 根据是否传入残差选择普通 RMSNorm 或融合的 Add+RMSNorm 路径。
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 没有残差时直接归一化当前输入。
        if residual is None:
            # 调用只做 RMSNorm 的已编译函数。
            return self.rms_forward(x)
        # 有残差时使用融合路径，同时生成更新后的残差。
        else:
            # 调用 Add+RMSNorm 的已编译函数。
            return self.add_rms_forward(x, residual)
