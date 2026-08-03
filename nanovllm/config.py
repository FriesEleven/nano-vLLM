# 导入文件系统工具，用于验证模型目录是否存在。
import os
# 导入数据类装饰器，用声明式字段定义配置对象。
from dataclasses import dataclass
# 导入 Hugging Face 自动配置加载器。
from transformers import AutoConfig


# 将类声明为使用 slots 的数据类，减少配置对象的属性开销。
@dataclass(slots=True)
# 定义贯穿推理引擎的模型与运行时配置。
class Config:
    # 保存本地 Hugging Face 模型目录路径。
    model: str
    # 限制一个调度批次可容纳的最大 Token 总数。
    max_num_batched_tokens: int = 16384
    # 限制一个调度批次可同时容纳的最大请求数。
    max_num_seqs: int = 512
    # 限制模型可处理的最大上下文长度。
    max_model_len: int = 4096
    # 指定可用于 KV Cache 等运行时状态的 GPU 显存比例。
    gpu_memory_utilization: float = 0.9
    # 指定张量并行使用的 GPU 数量。
    tensor_parallel_size: int = 1
    # 指示是否禁用 CUDA Graph 等图执行优化而强制即时执行。
    enforce_eager: bool = False
    # 缓存从模型目录读取到的 Hugging Face 配置对象。
    hf_config: AutoConfig | None = None
    # 保存模型终止符 Token ID，初始值表示尚未设置。
    eos: int = -1
    # 指定每个 Paged KV Cache 物理块可容纳的 Token 数。
    kvcache_block_size: int = 256
    # 保存可分配的 KV Cache 块数，负值表示由运行时计算。
    num_kvcache_blocks: int = -1

    # 在数据类初始化字段后验证输入并加载模型配置。
    def __post_init__(self):
        # 确保用户提供的模型路径确实是本地目录。
        assert os.path.isdir(self.model)
        # 确保 KV Cache 块大小满足底层实现的 256 Token 对齐要求。
        assert self.kvcache_block_size % 256 == 0
        # 将张量并行度限制在当前实现支持的 1 到 8 卡范围内。
        assert 1 <= self.tensor_parallel_size <= 8
        # 从模型目录加载 Hugging Face 配置以获取模型结构参数。
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 取用户设置与模型位置编码上限中的较小值，防止上下文越界。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
