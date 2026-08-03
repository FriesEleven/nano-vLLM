# 使用双端队列实现请求的等待队列与运行队列。
from collections import deque

# 导入调度器所需的批处理、KV Cache 和 EOS 配置。
from nanovllm.config import Config
# 导入请求状态对象及其生命周期枚举。
from nanovllm.engine.sequence import Sequence, SequenceStatus
# 导入负责物理 KV Cache Block 分配和前缀缓存的管理器。
from nanovllm.engine.block_manager import BlockManager


# 实现 Prefill 优先、Continuous Batching、Decode 和抢占的请求调度器。
class Scheduler:

    # 从配置初始化批次上限、结束 Token 和底层 BlockManager。
    def __init__(self, config: Config):
        # 保存每轮最多并行调度的序列数。
        self.max_num_seqs = config.max_num_seqs
        # 保存每轮 Prefill 可处理的 Token 总预算。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 保存分词器定义的 EOS Token ID，用于结束检测。
        self.eos = config.eos
        # 保存逻辑 KV Cache Block 大小，供缓存 Token 计算使用。
        self.block_size = config.kvcache_block_size
        # 创建管理固定显存 Block 的分配器和前缀缓存索引。
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 保存尚未完成 Prefill 或被抢占后等待重算的序列。
        self.waiting: deque[Sequence] = deque()
        # 保存已经完成 Prefill、可逐 Token Decode 的序列。
        self.running: deque[Sequence] = deque()

    # 判断是否没有任何等待或运行中的请求。
    def is_finished(self):
        # 两个队列均为空时，所有请求都已完成。
        return not self.waiting and not self.running

    # 将新建请求追加到等待队列尾部，维持到达顺序。
    def add(self, seq: Sequence):
        # 入队等待 KV Cache 分配与 Prefill 调度。
        self.waiting.append(seq)

    # 选择本轮执行的请求，优先安排 Prefill；若没有则安排 Decode。
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 初始化本轮将被送入 ModelRunner 的序列列表。
        scheduled_seqs = []
        # 初始化本轮已消耗的 Prefill Token 预算。
        num_batched_tokens = 0

        # prefill：优先尽可能处理等待请求，以加快首 Token 产生。
        # 等待队列非空且未达到并行序列上限时持续选取请求。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 查看队首请求，但确认可调度前不将其移出队列。
            seq = self.waiting[0]
            # 计算本轮还能容纳多少 Prefill Token。
            remaining = self.max_num_batched_tokens - num_batched_tokens
            # Token 预算用尽时不能再加入任何 Prefill 请求。
            if remaining == 0:
                # 结束 Prefill 选择，转而返回已有批次。
                break
            # 首次调度该请求时，需要先评估并建立 KV Cache Block 映射。
            if not seq.block_table:
                # 查询可连续复用的前缀 Block 数，并验证空闲 Block 是否足够。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # -1 表示物理 KV Cache 空间不足。
                if num_cached_blocks == -1:
                    # 保持队首不动，等待 Decode 释放或抢占腾出空间后再试。
                    break
                # 未命中缓存的 Token 才需要执行 Prefill。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            # 已经分配过 Block Table，说明这是 Chunked Prefill 的后续分块。
            else:
                # 只计算此前尚未提交到 KV Cache 的 Token。
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 只有本轮第一个请求允许被切分为 Chunked Prefill。
            if remaining < num_tokens and scheduled_seqs:
                # 后续请求不能挤占不足的 Token 预算，保留到下一轮处理。
                break
            # 仅首次接纳请求时真正占用或引用对应的物理 Block。
            if not seq.block_table:
                # 建立 Block Table 并登记前缀缓存共享引用。
                self.block_manager.allocate(seq, num_cached_blocks)
            # 设置本轮实际执行量；首个长请求可被预算截断。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            # 累加本轮已经安排的 Prefill Token 数。
            num_batched_tokens += seq.num_scheduled_tokens
            # 当缓存加本轮计算已覆盖全提示词时，Prefill 即告完成。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 更新状态，使后续轮次进入 Decode 流程。
                seq.status = SequenceStatus.RUNNING
                # 从等待队列移除刚完成 Prefill 的队首序列。
                self.waiting.popleft()
                # 将该序列追加到运行队列尾部。
                self.running.append(seq)
            # 无论 Prefill 是否完整，均加入本轮模型执行批次。
            scheduled_seqs.append(seq)

        # 若已选择任何 Prefill 序列，就不在同一轮混入 Decode。
        if scheduled_seqs:
            # 返回 Prefill 批次以及阶段标志。
            return scheduled_seqs, True

        # decode：仅当没有 Prefill 可执行时，为运行序列各安排一个 Token。
        # 运行队列非空且未达到并行序列上限时持续选取 Decode 请求。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 取出队首运行序列，准备检查其尾部 Block 容量。
            seq = self.running.popleft()
            # 当追加下一个 Token 会需要但没有空闲 Block 时触发抢占。
            while not self.block_manager.can_append(seq):
                # 优先抢占当前队列末尾的其他请求，尽量让当前序列继续前进。
                if self.running:
                    # 释放末尾请求的 Block，并将其放回等待队列重新 Prefill。
                    self.preempt(self.running.pop())
                # 没有其他运行请求可牺牲时，只能抢占当前取出的序列。
                else:
                    # 释放当前序列的 Block，并恢复其等待状态。
                    self.preempt(seq)
                    # 跳出内层循环，避免将已抢占的序列加入本轮 Decode。
                    break
            # 当内层 while 未经 break 结束时，说明可安全追加一个 Token。
            else:
                # Decode 阶段每条序列每轮只执行一个 Token。
                seq.num_scheduled_tokens = 1
                # 标记序列已离开 Prefill，用于跨进程序列化优化。
                seq.is_prefill = False
                # 若该 Token 跨越了 Block 边界，则分配新的尾部物理 Block。
                self.block_manager.may_append(seq)
                # 将可执行的 Decode 序列加入本轮批次。
                scheduled_seqs.append(seq)
        # 若无 Prefill 且也无 Decode 可调度，则内部状态不符合预期。
        assert scheduled_seqs
        # 将已调度序列按原先顺序放回运行队首，等待 postprocess 后继续调度。
        self.running.extendleft(reversed(scheduled_seqs))
        # 返回 Decode 批次以及阶段标志。
        return scheduled_seqs, False

    # 将一个运行请求抢占回等待队列，并释放其所有 KV Cache Block。
    def preempt(self, seq: Sequence):
        # 将生命周期改回等待状态。
        seq.status = SequenceStatus.WAITING
        # 标记后续需重新执行 Prefill，而非直接 Decode。
        seq.is_prefill = True
        # 释放该序列持有或共享的所有物理 KV Cache Block 引用。
        self.block_manager.deallocate(seq)
        # 放回等待队首，以便尽快被重新分配和重算。
        self.waiting.appendleft(seq)

    # 消费模型采样结果，提交本轮 KV Cache 进度并处理完成与回收。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # 按模型输出顺序配对每条执行序列及其新采样 Token。
        for seq, token_id in zip(seqs, token_ids):
            # 将本轮首次填满的 Block 登记为可复用前缀缓存。
            self.block_manager.hash_blocks(seq)
            # 将本轮模型已写入 KV Cache 的 Token 计入缓存进度。
            seq.num_cached_tokens += seq.num_scheduled_tokens
            # 清除本轮调度量，等待下一轮重新设置。
            seq.num_scheduled_tokens = 0
            # 分块 Prefill 尚未覆盖完整提示词时不应采样或追加 Token。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # 保持序列等待下一段 Prefill。
                continue
            # 将模型对当前上下文预测出的 Token 追加到序列末尾。
            seq.append_token(token_id)
            # 遇 EOS（且未忽略）或达到补全上限时结束请求。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 将序列标记为完成，以便引擎返回结果。
                seq.status = SequenceStatus.FINISHED
                # 释放其 KV Cache Block；完整 Block 仍可作为前缀缓存保留。
                self.block_manager.deallocate(seq)
                # 从运行队列移除已完成序列。
                self.running.remove(seq)
