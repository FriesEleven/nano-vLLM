# 提供张量运算和模型执行所需的 PyTorch API。
import torch
# 引入神经网络模块基类及层容器。
from torch import nn
# 引入张量并行所需的分布式通信接口。
import torch.distributed as dist
# 引入 Hugging Face 定义的 Qwen3 配置类型。
from transformers import Qwen3Config

# 引入融合的 SiLU 门控激活层。
from nanovllm.layers.activation import SiluAndMul
# 引入负责 FlashAttention 与 KV Cache 的注意力层。
from nanovllm.layers.attention import Attention
# 引入 Qwen3 使用的 RMSNorm 归一化层。
from nanovllm.layers.layernorm import RMSNorm
# 引入张量并行线性层。
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
# 引入按配置构造并缓存 RoPE 的工厂函数。
from nanovllm.layers.rotary_embedding import get_rope
# 引入词表并行的嵌入层和输出头。
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# 定义一个 Qwen3 解码器层中的自注意力子模块。
class Qwen3Attention(nn.Module):

    # 根据模型配置和张量并行规模初始化注意力投影、RoPE 与注意力计算层。
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        # 初始化 nn.Module 的内部状态和子模块注册机制。
        super().__init__()
        # 获取当前张量并行进程组中的总 rank 数。
        tp_size = dist.get_world_size()
        # 保存全局 Query 注意力头数。
        self.total_num_heads = num_heads
        # 要求 Query 头能被各个 tensor-parallel rank 均分。
        assert self.total_num_heads % tp_size == 0
        # 计算本 rank 持有的 Query 头数。
        self.num_heads = self.total_num_heads // tp_size
        # 保存全局 Key/Value 注意力头数。
        self.total_num_kv_heads = num_kv_heads
        # 要求 KV 头同样能被各 rank 均分。
        assert self.total_num_kv_heads % tp_size == 0
        # 计算本 rank 持有的 KV 头数。
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # 使用显式 head_dim，或由隐藏维度均分得到。
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 计算本 rank Query 投影输出的总维度。
        self.q_size = self.num_heads * self.head_dim
        # 计算本 rank 单个 Key 或 Value 投影的总维度。
        self.kv_size = self.num_kv_heads * self.head_dim
        # 计算缩放点积注意力使用的 1/sqrt(head_dim)。
        self.scaling = self.head_dim ** -0.5
        # 记录 QKV 投影是否含偏置，并决定是否启用 QK-Norm。
        self.qkv_bias = qkv_bias

        # 创建把分离的 Q/K/V 权重加载为一份本地分片的列并行投影层。
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 创建将各 rank 注意力输出相加还原为隐藏维度的行并行输出投影。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 仅在缩放配置是字典时读取其可能覆盖默认值的 rope_theta。
        if isinstance(rope_scaling, dict):
            # 优先使用缩放配置指定的旋转位置编码基数。
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # 创建或复用匹配当前头维度和最大位置的旋转位置编码模块。
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 创建实际执行 Prefill/Decode 注意力计算的通用注意力层。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 无 QKV 偏置的 Qwen3 变体使用逐头 QK RMSNorm。
        if not self.qkv_bias:
            # 为 Query 向量构造 RMSNorm。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            # 为 Key 向量构造 RMSNorm。
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    # 将隐藏状态投影为 QKV，施加位置编码后计算因果自注意力并输出投影结果。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 对输入隐藏状态执行合并的本地 QKV 投影。
        qkv = self.qkv_proj(hidden_states)
        # 按本 rank 的 Q/K/V 维度切分投影结果。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 将 Query 重排为 [token 数, 本地头数, 头维度]。
        q = q.view(-1, self.num_heads, self.head_dim)
        # 将 Key 重排为按 KV 头组织的三维张量。
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        # 将 Value 重排为按 KV 头组织的三维张量。
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 仅无偏置配置需要对 Q 与 K 做额外的 RMS 归一化。
        if not self.qkv_bias:
            # 归一化每个 Query 头的向量幅度。
            q = self.q_norm(q)
            # 归一化每个 Key 头的向量幅度。
            k = self.k_norm(k)
        # 为 Query 和 Key 注入 token 的相对位置信息。
        q, k = self.rotary_emb(positions, q, k)
        # 根据当前运行上下文执行 Prefill 或 Decode 因果注意力。
        o = self.attn(q, k, v)
        # 合并头维度并通过行并行层聚合为隐藏状态。
        output = self.o_proj(o.flatten(1, -1))
        # 返回与输入 token 对齐的注意力子层输出。
        return output


# 定义 Qwen3 解码器层中的 SwiGLU 前馈网络子模块。
class Qwen3MLP(nn.Module):

    # 初始化门控上投影、下投影和所用的 SiLU 激活函数。
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        # 初始化模块基类，以便注册下面的参数和子模块。
        super().__init__()
        # 合并 gate_proj 与 up_proj，减少一次输入读取和一次矩阵乘调用。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 将本 rank 中间层分片投影回隐藏维度并跨 rank 规约求和。
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 当前实现仅支持 Qwen3 的 SiLU 门控激活。
        assert hidden_act == "silu"
        # 创建融合 SiLU(gate) * up 的 SwiGLU 激活层。
        self.act_fn = SiluAndMul()

    # 执行合并上投影、门控激活和下投影，得到前馈网络输出。
    def forward(self, x):
        # 一次列并行线性层同时生成 gate 与 up 两部分。
        gate_up = self.gate_up_proj(x)
        # 对前半部分做 SiLU 后与后半部分逐元素相乘。
        x = self.act_fn(gate_up)
        # 将激活后的中间状态通过行并行层投影回隐藏维度。
        x = self.down_proj(x)
        # 返回 MLP 子层输出。
        return x


# 定义由自注意力、MLP、两个归一化和残差连接构成的一个 Qwen3 Decoder 层。
class Qwen3DecoderLayer(nn.Module):

    # 根据完整 Qwen3 配置创建这一层的全部子模块。
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        # 初始化模块基类。
        super().__init__()
        # 构造自注意力层，并从配置中读取模型结构与 RoPE 参数。
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # 构造 attention 后的 SwiGLU 前馈网络。
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 创建 attention 前的残差 RMSNorm。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 创建 MLP 前的残差 RMSNorm。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 使用融合的残差加法/RMSNorm 依次执行 attention 和 MLP 子层。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第一层没有前层输出的残差，因此直接归一化当前嵌入并保存其为残差。
        if residual is None:
            # 产生归一化输入，同时保留未归一化输入作残差。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        # 后续层把上一子层输出与延迟传递的残差相加再归一化。
        else:
            # 融合执行残差相加和 attention 前 RMSNorm。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 计算当前层的自注意力输出。
        hidden_states = self.self_attn(positions, hidden_states)
        # 融合残差加法并为 MLP 准备归一化输入。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 计算当前层的门控前馈输出。
        hidden_states = self.mlp(hidden_states)
        # 返回延迟相加的子层输出及其累计残差。
        return hidden_states, residual


# 定义去除语言模型输出头后的 Qwen3 主干网络。
class Qwen3Model(nn.Module):

    # 创建词嵌入、所有 Decoder 层和末尾 RMSNorm。
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        # 初始化模块基类。
        super().__init__()
        # 创建按词表分片的输入嵌入表。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 按配置深度创建并注册所有 Decoder 层。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建主干输出前的最终 RMSNorm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 将 token id 依次通过词嵌入、全部 Decoder 层和最终归一化。
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 将输入 token id 查询为本地并行嵌入后的隐藏状态。
        hidden_states = self.embed_tokens(input_ids)
        # 首层尚无需要累积的残差张量。
        residual = None
        # 按深度顺序执行所有 Decoder 层，并在线程中传递延迟残差。
        for layer in self.layers:
            # 计算一层 Transformer 并更新隐藏状态与残差。
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 将最后一个子层输出与残差融合后执行最终 RMSNorm。
        hidden_states, _ = self.norm(hidden_states, residual)
        # 返回可供语言模型头映射的最终隐藏状态。
        return hidden_states


# 定义带词表输出头的完整 Qwen3 因果语言模型。
class Qwen3ForCausalLM(nn.Module):
    # 描述 Hugging Face 拆分权重到本实现合并线性层的加载映射。
    packed_modules_mapping = {
        # 将 Query 投影权重装入合并 QKV 层的 q 分片。
        "q_proj": ("qkv_proj", "q"),
        # 将 Key 投影权重装入合并 QKV 层的 k 分片。
        "k_proj": ("qkv_proj", "k"),
        # 将 Value 投影权重装入合并 QKV 层的 v 分片。
        "v_proj": ("qkv_proj", "v"),
        # 将门控投影权重装入合并 MLP 层的第 0 段。
        "gate_proj": ("gate_up_proj", 0),
        # 将上投影权重装入合并 MLP 层的第 1 段。
        "up_proj": ("gate_up_proj", 1),
    }

    # 根据配置创建 Qwen3 主干和词表并行的语言模型输出头。
    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        # 初始化模块基类。
        super().__init__()
        # 创建输入到最终隐藏状态的 Transformer 主干。
        self.model = Qwen3Model(config)
        # 创建把隐藏状态映射为词表 logits 的并行输出头。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 配置要求权重共享时，让输出头直接复用输入嵌入表的参数存储。
        if config.tie_word_embeddings:
            # 绑定输入 embedding 与输出 head 的权重数据。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # 前向计算主干隐藏状态；logits 在调度器需要时通过 compute_logits 单独计算。
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 将 token id 和位置交给 Transformer 主干处理。
        return self.model(input_ids, positions)

    # 把最终隐藏状态投影到完整词表的未归一化分数。
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 调用词表并行输出头生成 logits。
        return self.lm_head(hidden_states)
