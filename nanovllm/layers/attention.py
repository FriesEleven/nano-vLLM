# 提供张量类型和缓存张量操作。
import torch
# 引入注意力层所继承的模块基类。
from torch import nn
# 引入 Triton JIT 编译器。
import triton
# 引入在 Triton 内核中可用的张量原语。
import triton.language as tl

# 引入变长 Prefill 与带 KV Cache Decode 的 FlashAttention 实现。
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
# 引入由调度器写入的当前批次运行上下文。
from nanovllm.utils.context import get_context


# 将 KV 写入逻辑即时编译为在 GPU 上并行执行的 Triton 内核。
@triton.jit
# 将一个 token 的 Key/Value 向量写入其 slot_mapping 指定的物理缓存槽位。
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # 取得本 Triton program 对应的扁平 token 下标。
    idx = tl.program_id(0)
    # 读取调度器为该 token 分配的物理 KV 槽位。
    slot = tl.load(slot_mapping_ptr + idx)
    # -1 表示该 token 不需要写缓存，直接结束当前 program。
    if slot == -1: return
    # 计算源 Key 向量 D 个元素的内存偏移。
    key_offsets = idx * key_stride + tl.arange(0, D)
    # 计算源 Value 向量 D 个元素的内存偏移。
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 从连续的当前批次 Key 张量加载一个 token 的所有头维度。
    key = tl.load(key_ptr + key_offsets)
    # 从连续的当前批次 Value 张量加载一个 token 的所有头维度。
    value = tl.load(value_ptr + value_offsets)
    # 计算目标物理缓存槽位内各元素的偏移。
    cache_offsets = slot * D + tl.arange(0, D)
    # 将 Key 向量写入分页 KV Cache。
    tl.store(k_cache_ptr + cache_offsets, key)
    # 将 Value 向量写入分页 KV Cache。
    tl.store(v_cache_ptr + cache_offsets, value)


# 检查张量布局后启动 Triton 内核，把当前批次的 KV 写入分页缓存。
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # 读取当前扁平 token 数、本地 KV 头数和每头维度。
    N, num_heads, head_dim = key.shape
    # 计算一个 token 的完整 Key/Value 向量长度。
    D = num_heads * head_dim
    # 要求每个头向量最后一维连续，满足内核连续加载假设。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # 要求相邻注意力头之间恰好跨过一个头维度。
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # 要求缓存的相邻 slot 之间恰好跨过一个完整 KV 向量。
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # 要求每个当前 token 都有一个对应的缓存槽位映射。
    assert slot_mapping.numel() == N
    # 为每个 token 启动一个 program 执行 KV 写入。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# 定义复用调度上下文、KV Cache 和 FlashAttention 的注意力计算层。
class Attention(nn.Module):

    # 记录本 rank 注意力形状和缩放系数，并初始化空的缓存引用。
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        # 初始化模块基类。
        super().__init__()
        # 保存本 rank 的 Query 头数。
        self.num_heads = num_heads
        # 保存单个注意力头的通道数。
        self.head_dim = head_dim
        # 保存 softmax 前的缩放点积系数。
        self.scale = scale
        # 保存本 rank 的 Key/Value 头数。
        self.num_kv_heads = num_kv_heads
        # 先以空张量占位，运行器稍后注入共享的 K/V 缓存。
        self.k_cache = self.v_cache = torch.tensor([])

    # 依据上下文选择 Prefill 或 Decode 的因果注意力内核，并返回每个 token 的注意力输出。
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 获取调度器为当前批次设置的位置、长度、slot 与块表信息。
        context = get_context()
        # 取出运行器注入的 Key 与 Value 分页缓存。
        k_cache, v_cache = self.k_cache, self.v_cache
        # 只有运行器已分配缓存时，才把本批新生成的 KV 写入对应物理槽位。
        if k_cache.numel() and v_cache.numel():
            # 按 slot mapping 将新 KV 写入缓存。
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # Prefill 一次计算多个变长序列中所有输入 token 的因果注意力。
        if context.is_prefill:
            # 有块表意味着命中了前缀缓存，FlashAttention 应从全局缓存读取 K/V。
            # prefix cache
            if context.block_tables is not None:
                # 用分页缓存替换当前批次局部 KV 作为注意力的键和值来源。
                k, v = k_cache, v_cache
            # 调用支持变长序列和分页块表的 Prefill FlashAttention。
            o = flash_attn_varlen_func(q, k, v,
                                       # 传入 Query 的最大长度和累积长度偏移。
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       # 传入 Key/Value 的最大长度和累积长度偏移。
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       # 指定缩放因子、因果掩码与可选分页块表。
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        # Decode 每个请求通常只有一个新 token，但需关注完整历史缓存。
        # decode
        else:
            # 增加长度维度后调用直接读取 KV Cache 的 Decode 内核。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        # 传入各请求已缓存长度及逻辑块到物理块映射。
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        # 保持与 Prefill 一致的缩放和因果约束。
                                        softmax_scale=self.scale, causal=True)
        # 返回 FlashAttention 生成的多头注意力结果。
        return o
