# 提供张量、参数和临时输出缓冲区操作。
import torch
# 引入模块基类和可训练参数封装。
from torch import nn
# 引入嵌入查询与线性投影函数。
import torch.nn.functional as F
# 引入词表张量并行所需的集合通信操作。
import torch.distributed as dist

# 引入用于识别 Prefill/Decode 批次的调度上下文。
from nanovllm.utils.context import get_context


# 定义按词表维度切分权重、通过 all-reduce 合并查询结果的嵌入层。
class VocabParallelEmbedding(nn.Module):

    # 根据当前张量并行 rank 创建本地词表分片及其权重加载器。
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        # 初始化模块基类。
        super().__init__()
        # 获取当前进程在张量并行组中的 rank。
        self.tp_rank = dist.get_rank()
        # 获取词表切分所使用的总 rank 数。
        self.tp_size = dist.get_world_size()
        # 要求完整词表大小能被各 rank 均匀切分。
        assert num_embeddings % self.tp_size == 0
        # 保存全局词表大小。
        self.num_embeddings = num_embeddings
        # 计算每个 rank 保存的词条数。
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 计算本 rank 词表分片的起始全局 id。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 计算本 rank 词表分片的结束（开区间）全局 id。
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 分配本地词条到嵌入向量的可训练权重。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 让统一权重加载器能够调用本层的分片加载逻辑。
        self.weight.weight_loader = self.weight_loader

    # 从完整 checkpoint 词表权重中截取当前 rank 对应的连续词表行。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 取得不参与 autograd 跟踪的目标参数数据。
        param_data = param.data
        # 读取本地词表分片包含的行数。
        shard_size = param_data.size(0)
        # 计算本 rank 在完整词表中的起始行号。
        start_idx = self.tp_rank * shard_size
        # 沿词表行维度切出当前 rank 的权重分片。
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 将 checkpoint 分片原位复制到本地参数。
        param_data.copy_(loaded_weight)

    # 查询输入 token 的嵌入；多卡时由拥有该 token 的 rank 贡献非零结果。
    def forward(self, x: torch.Tensor):
        # 多卡时先将全局 token id 映射到本地词表 id，并标记哪些 id 属于本分片。
        if self.tp_size > 1:
            # 生成本 rank 是否拥有每个 token 的布尔掩码。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将拥有的全局 id 转为本地 id；其他 id 安全地映射为 0。
            x = mask * (x - self.vocab_start_idx)
        # 从本地嵌入表查询所有（本地化后的）token 向量。
        y = F.embedding(x, self.weight)
        # 多卡时屏蔽非本分片 token 的伪查询结果，并把各 rank 的结果逐元素相加。
        if self.tp_size > 1:
            # 将 token 掩码扩展到嵌入维度，清零不属于本 rank 的行。
            y = mask.unsqueeze(1) * y
            # 汇总所有 rank 的局部嵌入，使每个 rank 得到完整嵌入结果。
            dist.all_reduce(y)
        # 返回与输入 token 形状对应的嵌入向量。
        return y


# 定义复用词表分片权重布局、但输出完整 logits 的并行语言模型头。
class ParallelLMHead(VocabParallelEmbedding):

    # 初始化不带偏置的词表并行输出头。
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        # 当前实现和 Qwen3 权重格式均不支持 LM Head 偏置。
        assert not bias
        # 复用词表分片权重及加载逻辑。
        super().__init__(num_embeddings, embedding_dim)

    # 将隐藏状态映射为词表 logits；Prefill 时仅保留每个请求的最后一个 token。
    def forward(self, x: torch.Tensor):
        # 获取当前批次是否为 Prefill 及其累积序列长度。
        context = get_context()
        # Prefill 的目标是为每个序列采样下一个 token，因此只需计算每段的尾部隐藏状态。
        if context.is_prefill:
            # 根据累积 Query 长度计算每条请求最后一个 token 的扁平下标。
            last_indices = context.cu_seqlens_q[1:] - 1
            # 抽取尾部隐藏状态并确保其内存连续，便于线性层执行。
            x = x[last_indices].contiguous()
        # 用本地词表分片权重生成局部词表 logits。
        logits = F.linear(x, self.weight)
        # 多卡时将每个 rank 的词表维度分片收集到 rank 0 并按最后一维拼接。
        if self.tp_size > 1:
            # 仅 rank 0 预分配接收各分片 logits 的列表。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 将所有 rank 的局部 logits 汇聚到 rank 0。
            dist.gather(logits, all_logits, 0)
            # rank 0 拼成全词表 logits，其他 rank 不保留结果。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        # 返回 rank 0 的完整 logits，或单卡时的本地（即完整）logits。
        return logits
