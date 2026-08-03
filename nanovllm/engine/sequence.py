# 使用浅拷贝避免调用方随后修改输入列表影响序列内容。
from copy import copy
# 用枚举表达请求生命周期，并自动生成各状态的唯一值。
from enum import Enum, auto
# 提供递增计数器，为每个请求分配唯一序列 ID。
from itertools import count

# 导入采样配置，初始化生成温度与停止条件。
from nanovllm.sampling_params import SamplingParams


# 定义请求序列在调度器中的三个生命周期状态。
class SequenceStatus(Enum):
    # 等待 Block 分配和 Prefill 调度。
    WAITING = auto()
    # 已进入 Decode 队列，持续生成下一个 Token。
    RUNNING = auto()
    # 已遇到停止条件，结果可返回给调用者。
    FINISHED = auto()


# 保存单个生成请求的 Token、调度进度和 KV Cache Block 映射。
class Sequence:
    # 默认每个逻辑 KV Cache Block 覆盖 256 个 Token，后续由引擎配置覆盖。
    block_size = 256
    # 在进程内持续递增的序列 ID 生成器。
    counter = count()

    # 用提示词 Token 和采样参数创建一个等待执行的生成序列。
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 取出下一个唯一 ID，以便结果按请求区分。
        self.seq_id = next(Sequence.counter)
        # 新建请求需先进入等待队列。
        self.status = SequenceStatus.WAITING
        # 拷贝提示词 Token，避免共享调用方列表。
        self.token_ids = copy(token_ids)
        # 保存当前最后一个 Token，Decode 阶段只需输入它。
        self.last_token = token_ids[-1]
        # 记录提示词与已生成 Token 的总长度。
        self.num_tokens = len(self.token_ids)
        # 固定记录原始提示词长度，用于拆分输出。
        self.num_prompt_tokens = len(token_ids)
        # 初始没有可复用或已提交到 KV Cache 的 Token。
        self.num_cached_tokens = 0
        # 初始尚未有 Token 被本轮调度给模型计算。
        self.num_scheduled_tokens = 0
        # 首轮需要处理完整提示词，之后才转入 Decode。
        self.is_prefill = True
        # 初始尚未建立逻辑 Block 到物理 KV Cache Block 的映射。
        self.block_table = []
        # 保存该序列独立的采样温度。
        self.temperature = sampling_params.temperature
        # 保存最多可生成的补全文本 Token 数。
        self.max_tokens = sampling_params.max_tokens
        # 保存是否忽略模型输出的 EOS Token。
        self.ignore_eos = sampling_params.ignore_eos

    # 使 len(seq) 返回当前序列的 Token 总数。
    def __len__(self):
        # 委托给维护好的长度计数，避免每次遍历列表。
        return self.num_tokens

    # 使序列可通过下标或切片访问其 Token ID。
    def __getitem__(self, key):
        # 直接复用内部 Token 列表的索引语义。
        return self.token_ids[key]

    # 将“是否已完成”暴露为只读派生属性。
    @property
    def is_finished(self):
        # 仅在状态机进入 FINISHED 时返回真。
        return self.status == SequenceStatus.FINISHED

    # 计算提示词之后已经生成的 Token 数量。
    @property
    def num_completion_tokens(self):
        # 总长度减去固定提示词长度即为补全长度。
        return self.num_tokens - self.num_prompt_tokens

    # 返回不含补全内容的原始提示词 Token 切片。
    @property
    def prompt_token_ids(self):
        # 从开头截取到提示词边界。
        return self.token_ids[:self.num_prompt_tokens]

    # 返回提示词之后的所有已生成 Token 切片。
    @property
    def completion_token_ids(self):
        # 从提示词边界截取到当前末尾。
        return self.token_ids[self.num_prompt_tokens:]

    # 计算当前 Token 序列需要多少个固定大小的逻辑 Block。
    @property
    def num_blocks(self):
        # 通过向上取整覆盖尾部不满的 Block。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 计算最后一个逻辑 Block 中实际有效的 Token 数。
    @property
    def last_block_num_tokens(self):
        # 用总长度减去此前完整 Block 的容量。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回第 i 个逻辑 Block 对应的 Token 切片。
    def block(self, i):
        # 防止访问不存在的逻辑 Block。
        assert 0 <= i < self.num_blocks
        # 依据固定 Block 大小计算切片边界。
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 将采样得到的新 Token 追加到序列末尾并同步维护长度状态。
    def append_token(self, token_id: int):
        # 把新生成 Token 写入完整 Token 历史。
        self.token_ids.append(token_id)
        # 更新 Decode 阶段下一轮所需的单 Token 输入。
        self.last_token = token_id
        # 维护总长度，避免每次调用 len(list)。
        self.num_tokens += 1

    # 定义进程间序列化时保留的精简状态，减少传给 Tensor Parallel 从进程的数据量。
    def __getstate__(self):
        # Decode 仅发送最后一个 Token，Prefill 则需发送完整 Token 列表。
        last_state = self.last_token if not self.is_prefill else self.token_ids
        # 返回重建运行所需的最小状态元组。
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # 从 __getstate__ 生成的精简状态恢复序列对象。
    def __setstate__(self, state):
        # 解包调度与 KV Cache 所需字段。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        # 列表表示该序列正处于 Prefill，需要完整 Token。
        if isinstance(last_state, list):
            # 直接恢复传输过来的完整 Token 列表。
            self.token_ids = last_state
            # 从完整列表恢复最后一个 Token。
            self.last_token = self.token_ids[-1]
        # 标量表示 Decode 阶段只传输最后一个 Token。
        else:
            # 从进程不需要完整历史 Token，因此保留空列表节省传输。
            self.token_ids = []
            # 使用传输的标量继续下一次 Decode。
            self.last_token = last_state
