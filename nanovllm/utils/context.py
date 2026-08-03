# 导入数据类装饰器，用于声明注意力计算上下文。
from dataclasses import dataclass
# 导入 PyTorch，以标注上下文中保存的张量类型。
import torch


# 将上下文声明为使用 slots 的数据类，降低频繁切换时的对象开销。
@dataclass(slots=True)
# 定义供注意力层读取的当前调度批次元数据。
class Context:
    # 标记当前前向计算是否处于 Prefill 阶段。
    is_prefill: bool = False
    # 保存查询序列的累积长度数组，用于变长注意力核。
    cu_seqlens_q: torch.Tensor | None = None
    # 保存键值序列的累积长度数组，用于变长注意力核。
    cu_seqlens_k: torch.Tensor | None = None
    # 保存本批查询序列中的最大长度。
    max_seqlen_q: int = 0
    # 保存本批键值序列中的最大长度。
    max_seqlen_k: int = 0
    # 保存每个 Token 在 Paged KV Cache 中对应物理槽位的映射。
    slot_mapping: torch.Tensor | None = None
    # 保存各请求已有上下文的长度。
    context_lens: torch.Tensor | None = None
    # 保存各请求从逻辑块到物理 KV Cache 块的映射表。
    block_tables: torch.Tensor | None = None

# 初始化模块级默认上下文，供未设置批次信息时安全读取。
_CONTEXT = Context()

# 返回当前模块级共享上下文对象。
def get_context():
    # 将当前上下文直接交给调用方读取。
    return _CONTEXT

# 用一次调度批次的全部注意力元数据替换共享上下文。
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # 声明下方赋值操作修改的是模块级变量而非局部变量。
    global _CONTEXT
    # 按 Context 字段定义的顺序创建并安装新的批次上下文。
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

# 将共享上下文恢复为所有字段均为默认值的空状态。
def reset_context():
    # 声明下方赋值操作修改的是模块级变量而非局部变量。
    global _CONTEXT
    # 创建新的默认 Context，以清除上一批次留下的元数据引用。
    _CONTEXT = Context()
