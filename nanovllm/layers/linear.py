# 提供张量、参数和类型标注。
import torch
# 引入线性层继承的模块基类。
from torch import nn
# 引入无状态的线性投影函数。
import torch.nn.functional as F
# 引入张量并行分片与规约通信接口。
import torch.distributed as dist


# 安全执行整除，确保张量并行切分不会遗失任何维度。
def divide(numerator, denominator):
    # 要求被切分维度能被分母（通常为 TP world size）整除。
    assert numerator % denominator == 0
    # 返回每个 rank 获得的等长分片大小。
    return numerator // denominator


# 定义所有自定义线性层共享的参数、并行元数据和抽象前向接口。
class LinearBase(nn.Module):

    # 分配本地权重与可选偏置，并记录权重实际被切分的维度。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        # 初始化模块基类。
        super().__init__()
        # 记录 checkpoint 权重应沿哪一维切分；None 表示完整复制。
        self.tp_dim = tp_dim
        # 获取本进程在张量并行通信组中的 rank。
        self.tp_rank = dist.get_rank()
        # 获取参与张量并行的总进程数。
        self.tp_size = dist.get_world_size()
        # 分配当前 rank 本地持有的二维权重分片。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 注册子类实现的 checkpoint 分片加载回调。
        self.weight.weight_loader = self.weight_loader
        # 仅请求偏置时才分配并注册偏置参数。
        if bias:
            # 分配本地输出通道对应的偏置参数。
            self.bias = nn.Parameter(torch.empty(output_size))
            # 偏置也复用当前线性层的加载逻辑。
            self.bias.weight_loader = self.weight_loader
        # 无偏置时显式注册 None，保证模块状态字典接口一致。
        else:
            # 将 bias 注册为不存在的参数而不是普通属性。
            self.register_parameter("bias", None)

    # 规定具体线性层必须定义自身的前向计算和必要的通信策略。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基类没有可执行的具体线性变换。
        raise NotImplementedError


# 定义权重在所有张量并行 rank 上完整复制的普通线性层。
class ReplicatedLinear(LinearBase):

    # 创建不切分输入或输出维度的完整线性层。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 将完整矩阵大小交给基类分配。
        super().__init__(input_size, output_size, bias)

    # 把 checkpoint 中的完整参数直接复制到每个 rank 的复制副本。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 不做分片，原位拷贝完整权重或偏置。
        param.data.copy_(loaded_weight)

    # 对输入执行不含通信的完整线性投影。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用本地完整权重和可选偏置计算输出。
        return F.linear(x, self.weight, self.bias)


# 定义沿输出通道（权重第 0 维）切分的列并行线性层。
class ColumnParallelLinear(LinearBase):

    # 将输出维度均分到各 rank；每个 rank 接收完整输入并产生部分输出通道。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 读取当前张量并行组大小。
        tp_size = dist.get_world_size()
        # 分配输出行切分后的本地权重，tp_dim=0 表示按行加载。
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    # 从完整 checkpoint 参数中按输出通道截取本 rank 对应的分片。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 获取不追踪梯度的本地目标存储。
        param_data = param.data
        # 读取本地在输出维度上应装载的长度。
        shard_size = param_data.size(self.tp_dim)
        # 计算本 rank 分片在完整参数中的起始位置。
        start_idx = self.tp_rank * shard_size
        # 沿输出维度裁剪出对应 checkpoint 分片。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        # 原位复制分片数据。
        param_data.copy_(loaded_weight)

    # 用完整输入和本地输出行分片计算局部输出，无需立即通信。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 产生本 rank 拥有的那部分输出通道。
        return F.linear(x, self.weight, self.bias)


# 定义把多个列并行投影拼接在一个权重矩阵中的线性层，例如 MLP 的 gate/up。
class MergedColumnParallelLinear(ColumnParallelLinear):

    # 记录各合并段的原始大小，并按所有段的总输出维度构造列并行层。
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # 保存每个原始投影段的全局输出维度，用于分段加载。
        self.output_sizes = output_sizes
        # 以各段总和作为合并矩阵的输出维度。
        super().__init__(input_size, sum(output_sizes), bias)

    # 将一个独立 checkpoint 投影段装入合并权重中与其对应的本地子区间。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # 获取完整的本地合并参数存储。
        param_data = param.data
        # 计算该段在本地合并输出维度中的起始偏移。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 计算该段在每个 rank 上的分片长度。
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 视图定位到本地合并参数中的目标子段。
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从独立 checkpoint 段中取出当前 rank 的切片。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # 将分片复制进合并参数的正确位置。
        param_data.copy_(loaded_weight)


# 定义按输出维度切分、且把 Q/K/V 三份投影权重装入同一矩阵的列并行层。
class QKVParallelLinear(ColumnParallelLinear):

    # 根据全局注意力头数计算每个 Q/K/V 分段大小，并创建合并的列并行投影。
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        # 读取张量并行组大小。
        tp_size = dist.get_world_size()
        # 未指定 GQA/MQA KV 头数时退化为与 Query 头数相同。
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        # 保存每个注意力头的通道数。
        self.head_size = head_size
        # 计算当前 rank 持有的 Query 头数量。
        self.num_heads = divide(total_num_heads, tp_size)
        # 计算当前 rank 持有的 Key/Value 头数量。
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 计算全局拼接 QKV 投影的输出维度。
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        # 创建沿拼接输出维度切分的列并行层。
        super().__init__(hidden_size, output_size, bias)

    # 按 q/k/v 标识选择合并矩阵中的目标区间，并加载该投影的本地 checkpoint 分片。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 获取本地合并 QKV 参数的底层存储。
        param_data = param.data
        # 拒绝未知投影名称，防止权重写入错误区间。
        assert loaded_shard_id in ["q", "k", "v"]
        # Query 段位于拼接 QKV 输出的最前面。
        if loaded_shard_id == "q":
            # 计算本 rank 的 Query 输出维度。
            shard_size = self.num_heads * self.head_size
            # Query 段从本地合并权重的开头开始。
            shard_offset = 0
        # Key 段紧随 Query 段之后。
        elif loaded_shard_id == "k":
            # 计算本 rank 的 Key 输出维度。
            shard_size = self.num_kv_heads * self.head_size
            # 跳过本地 Query 段得到 Key 起点。
            shard_offset = self.num_heads * self.head_size
        # Value 段紧随 Key 段之后。
        else:
            # 计算本 rank 的 Value 输出维度。
            shard_size = self.num_kv_heads * self.head_size
            # 跳过本地 Query 和 Key 段得到 Value 起点。
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        # 将目标写入视图缩小到 q/k/v 对应区间。
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从该独立投影的 checkpoint 权重取当前 rank 分片。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # 原位写入本地合并 QKV 参数。
        param_data.copy_(loaded_weight)


# 定义沿输入通道（权重第 1 维）切分、并以 all-reduce 合并结果的行并行线性层。
class RowParallelLinear(LinearBase):

    # 将输入维度均分到各 rank，而每个 rank 都保留完整输出维度。
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 读取张量并行组大小。
        tp_size = dist.get_world_size()
        # 分配输入列切分后的本地权重，tp_dim=1 表示按列加载。
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    # 加载行并行权重的输入列分片；偏置则在每个 rank 完整复制。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 获取不追踪梯度的本地目标参数。
        param_data = param.data
        # 一维参数是输出偏置，不按输入维度切分。
        if param_data.ndim == 1:
            # 将完整偏置复制到当前 rank。
            param_data.copy_(loaded_weight)
            # 偏置加载完成，不再执行二维权重切分。
            return
        # 读取本地输入列分片长度。
        shard_size = param_data.size(self.tp_dim)
        # 计算当前 rank 在完整输入列维度的起始下标。
        start_idx = self.tp_rank * shard_size
        # 从完整权重沿输入维度切出本地列分片。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        # 原位复制本地权重分片。
        param_data.copy_(loaded_weight)

    # 对本 rank 的输入分片投影，并通过 all-reduce 累加得到完整输出。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅 rank 0 加偏置，避免 all-reduce 后偏置被重复累加。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 多卡时求和所有 rank 的局部矩阵乘结果。
        if self.tp_size > 1:
            # 将各输入分片对同一输出通道的贡献相加。
            dist.all_reduce(y)
        # 返回所有 rank 都一致的完整输出张量。
        return y
