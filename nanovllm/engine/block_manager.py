# 用双端队列按先进先出顺序管理空闲 KV Cache Block。
from collections import deque
# 提供快速、稳定的 64 位哈希，用于定位可复用的前缀 Block。
import xxhash
# 将 Token ID 列表转换为连续字节，以供哈希函数处理。
import numpy as np

# 导入请求序列对象，供 Block 分配逻辑读取其 Token 与 Block Table。
from nanovllm.engine.sequence import Sequence


# 表示 KV Cache 中一个固定大小的物理存储块及其前缀缓存元数据。
class Block:

    # 使用物理 Block ID 初始化引用计数和缓存匹配信息。
    def __init__(self, block_id):
        # 记录该对象在预分配 KV Cache 中的物理索引。
        self.block_id = block_id
        # 0 表示当前没有序列持有该 Block。
        self.ref_count = 0
        # -1 是尚未写入可复用完整前缀哈希的哨兵值。
        self.hash = -1
        # 保存该完整 Block 的 Token，用于防止纯哈希碰撞造成误复用。
        self.token_ids = []

    # 将已填满的 Block 标记为可通过给定哈希复用。
    def update(self, hash: int, token_ids: list[int]):
        # 保存包含此前缀链的 64 位累计哈希。
        self.hash = hash
        # 保存实际 Token 序列以二次验证哈希命中。
        self.token_ids = token_ids

    # 将一个刚分配的 Block 恢复到正在被单个序列使用的初始状态。
    def reset(self):
        # 分配后立刻由调用方序列持有一次引用。
        self.ref_count = 1
        # 新 Block 尚未形成完整前缀，因此清除旧哈希。
        self.hash = -1
        # 清除旧 Token 快照，避免被后续查询误判为命中。
        self.token_ids = []


# 统一管理 Paged KV Cache 的物理 Block、引用计数和前缀缓存索引。
class BlockManager:

    # 根据可用 Block 总数和单个 Block 的 Token 容量创建管理器。
    def __init__(self, num_blocks: int, block_size: int):
        # 保存每个 KV Cache Block 可容纳的 Token 数。
        self.block_size = block_size
        # 预先构造所有物理 Block 元数据对象。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 建立“完整前缀哈希 -> 物理 Block ID”的复用索引。
        self.hash_to_block_id: dict[int, int] = dict()
        # 将所有尚未使用的 Block ID 放入空闲队列。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 记录当前已被至少一个序列持有的 Block ID。
        self.used_block_ids: set[int] = set()

    # 为一个完整 Token Block 计算与此前缀关联的链式哈希。
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 创建新的 xxHash64 累加器。
        h = xxhash.xxh64()
        # 只有存在前一个完整 Block 时才把其哈希纳入链路。
        if prefix != -1:
            # 将前缀哈希编码为固定 8 字节的小端数据。
            h.update(prefix.to_bytes(8, "little"))
        # 将当前 Block 的 Token 字节加入累计哈希。
        h.update(np.array(token_ids).tobytes())
        # 返回最终无符号 64 位摘要值。
        return h.intdigest()

    # 从空闲队列取出一个物理 Block，并使其进入已使用状态。
    def _allocate_block(self) -> int:
        # 取出最早归还的空闲 Block ID。
        block_id = self.free_block_ids.popleft()
        # 获取该 ID 对应的元数据对象。
        block = self.blocks[block_id]
        # 空闲 Block 不应被任何序列引用。
        assert block.ref_count == 0
        # 若旧哈希索引仍指向该 Block，则先失效。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            # 删除即将被覆盖的前缀缓存索引。
            del self.hash_to_block_id[block.hash]
        # 初始化引用计数并清空旧的缓存匹配元数据。
        block.reset()
        # 标记该物理 Block 已被使用。
        self.used_block_ids.add(block_id)
        # 将新分配的物理 Block ID 交给调用方写入 Block Table。
        return block_id

    # 回收引用归零的物理 Block，但保留其完整前缀哈希以便将来复用。
    def _deallocate_block(self, block_id: int):
        # 只有最后一个引用释放后才允许归还空闲队列。
        assert self.blocks[block_id].ref_count == 0
        # 从活跃集合移除，表示当前没有序列占用它。
        self.used_block_ids.remove(block_id)
        # 将 ID 放回队尾，供后续请求再次分配。
        self.free_block_ids.append(block_id)

    # 评估序列可复用的完整前缀 Block 数，并检查剩余 Block 是否足以完成分配。
    def can_allocate(self, seq: Sequence) -> int:
        # 以无前缀哨兵初始化链式哈希。
        h = -1
        # 统计从序列开头连续命中的完整缓存 Block 数。
        num_cached_blocks = 0
        # 初始假设序列的全部 Block 都需要新的物理空间。
        num_new_blocks = seq.num_blocks
        # 最后一个可能不完整的 Block 不能参与前缀缓存复用。
        for i in range(seq.num_blocks - 1):
            # 取出第 i 个完整逻辑 Block 的 Token ID。
            token_ids = seq.block(i)
            # 计算包含此前缀的当前 Block 链式哈希。
            h = self.compute_hash(token_ids, h)
            # 查询该哈希是否已对应某个历史完整 Block。
            block_id = self.hash_to_block_id.get(h, -1)
            # 无索引或 Token 二次校验失败时停止连续命中。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 后续 Block 依赖此前缀，不能再被视为可复用。
                break
            # 记录当前完整 Block 可从前缀缓存取得。
            num_cached_blocks += 1
            # 已被其他活跃序列持有时无需额外占用空闲池。
            if block_id in self.used_block_ids:
                # 共享引用不会消耗新的物理 Block。
                num_new_blocks -= 1
        # 空闲 Block 少于尚需分配数量时无法安全接纳该序列。
        if len(self.free_block_ids) < num_new_blocks:
            # 用 -1 向调度器报告显存 Block 不足。
            return -1
        # 返回可直接复用的连续完整 Block 数量。
        return num_cached_blocks

    # 为序列建立逻辑到物理 Block 的映射，并连接已命中的前缀缓存。
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 仅允许为尚未拥有 Block Table 的序列执行初次分配。
        assert not seq.block_table
        # 从头开始重新计算前缀链，以查回已命中的物理 Block。
        h = -1
        # 先处理可以直接复用的完整前缀 Block。
        for i in range(num_cached_blocks):
            # 读取当前逻辑 Block 的 Token 用于重建哈希。
            token_ids = seq.block(i)
            # 得到该逻辑 Block 对应的链式哈希。
            h = self.compute_hash(token_ids, h)
            # 从前缀索引取得已缓存的物理 Block ID。
            block_id = self.hash_to_block_id[h]
            # 获取物理 Block 元数据以维护引用关系。
            block = self.blocks[block_id]
            # 若该 Block 已被活跃序列使用，则共享同一份 KV Cache。
            if block_id in self.used_block_ids:
                # 追加一个持有者引用。
                block.ref_count += 1
            # 若缓存 Block 此前空闲，需要重新激活而不是覆盖它。
            else:
                # 由当前序列成为唯一持有者。
                block.ref_count = 1
                # 从空闲队列中移除这个被重新使用的 Block。
                self.free_block_ids.remove(block_id)
                # 标记它重新进入活跃集合。
                self.used_block_ids.add(block_id)
            # 在序列的逻辑 Block Table 中记录物理映射。
            seq.block_table.append(block_id)
        # 为未命中的后缀和最后一个不完整 Block 分配新空间。
        for i in range(num_cached_blocks, seq.num_blocks):
            # 追加一个全新的物理 Block ID。
            seq.block_table.append(self._allocate_block())
        # 记录可跳过 Prefill 的已缓存 Token 总数。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 释放序列对所有物理 Block 的引用，并清空它的缓存映射状态。
    def deallocate(self, seq: Sequence):
        # 逆序遍历逻辑映射，逐个释放 Block 引用。
        for block_id in reversed(seq.block_table):
            # 取得当前物理 Block 的元数据。
            block = self.blocks[block_id]
            # 移除该序列持有的一次引用。
            block.ref_count -= 1
            # 只有没有任何共享者时才可归还物理空间。
            if block.ref_count == 0:
                # 将此 Block 放回空闲池。
                self._deallocate_block(block_id)
        # 清除序列此前累计的已缓存 Token 数。
        seq.num_cached_tokens = 0
        # 清空逻辑到物理 Block 的映射，表示需要重新分配。
        seq.block_table.clear()

    # 判断在序列追加一个 Token 后是否需要、且是否有能力分配新的尾部 Block。
    def can_append(self, seq: Sequence) -> bool:
        # 仅当追加后进入新 Block 时要求至少一个空闲 Block。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 在即将跨越 Block 边界时，按需为序列追加一个新的尾部物理 Block。
    def may_append(self, seq: Sequence):
        # 当前长度模 Block 大小为 1，说明新生成 Token 已位于新的逻辑 Block。
        if len(seq) % self.block_size == 1:
            # 为这个新逻辑 Block 分配并登记物理存储位置。
            seq.block_table.append(self._allocate_block())

    # 将本轮刚计算完成的完整 Block 写入前缀缓存索引。
    def hash_blocks(self, seq: Sequence):
        # 找到本轮调度开始处对应的完整逻辑 Block 下标。
        start = seq.num_cached_tokens // self.block_size
        # 找到本轮结束后可完整缓存的逻辑 Block 终点。
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 本轮未填满任何 Block 时没有可安全复用的 KV Cache。
        if start == end: return
        # 继承上一个完整 Block 的链式哈希，或从无前缀开始。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        # 逐个登记本轮首次变为完整的逻辑 Block。
        for i in range(start, end):
            # 根据 Block Table 找到其物理 Block 元数据。
            block = self.blocks[seq.block_table[i]]
            # 读取完整逻辑 Block 的 Token 作为匹配凭据。
            token_ids = seq.block(i)
            # 把当前 Token 纳入前缀链，计算唯一的累计哈希。
            h = self.compute_hash(token_ids, h)
            # 在物理 Block 上保存哈希和 Token 快照。
            block.update(h, token_ids)
            # 使后续具有相同前缀的序列能查到该 Block。
            self.hash_to_block_id[h] = block.block_id
