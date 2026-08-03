# 序列化主进程发给 Tensor Parallel 从进程的方法调用与参数。
import pickle
# 提供 CUDA 张量、模型执行和 CUDA Graph 能力。
import torch
# 提供 NCCL 多卡进程组和同步屏障。
import torch.distributed as dist
# 为进程间命令通知提供 Event 类型标注。
from multiprocessing.synchronize import Event
# 提供主从进程共享的小型命令缓冲区。
from multiprocessing.shared_memory import SharedMemory

# 导入运行时配置类型。
from nanovllm.config import Config
# 导入请求序列，供预热和输入准备逻辑使用。
from nanovllm.engine.sequence import Sequence
# 导入项目实现的 Qwen3 因果语言模型。
from nanovllm.models.qwen3 import Qwen3ForCausalLM
# 导入将 logits 转换为下一个 Token 的采样器。
from nanovllm.layers.sampler import Sampler
# 导入注意力层读取的批次上下文管理函数。
from nanovllm.utils.context import set_context, get_context, reset_context
# 导入从模型目录加载权重的工具函数。
from nanovllm.utils.loader import load_model


# 在单卡或 Tensor Parallel 的每个 rank 上执行模型推理并管理其 KV Cache。
class ModelRunner:

    # 初始化当前 rank 的模型、进程组、KV Cache 和可选 CUDA Graph。
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 保存全局推理配置，供后续方法读取。
        self.config = config
        # 取出 Hugging Face 模型配置，便于设置模型结构和数据类型。
        hf_config = config.hf_config
        # 保存 Paged KV Cache 中单个 Block 的 Token 容量。
        self.block_size = config.kvcache_block_size
        # 保存是否强制使用 eager 执行而禁用 CUDA Graph。
        self.enforce_eager = config.enforce_eager
        # 保存 Tensor Parallel 进程总数。
        self.world_size = config.tensor_parallel_size
        # 保存当前进程在 Tensor Parallel 组中的 rank。
        self.rank = rank
        # 保存主从进程之间用于命令唤醒的 Event 或 Event 列表。
        self.event = event

        # 建立基于 NCCL 的本机多卡通信组。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 将当前进程绑定到与 rank 对应的 GPU。
        torch.cuda.set_device(rank)
        # 保存调用前的全局默认浮点类型，以便结束后恢复。
        default_dtype = torch.get_default_dtype()
        # 让随后创建的模型参数默认采用模型要求的精度。
        torch.set_default_dtype(hf_config.dtype)
        # 让随后未指定设备的张量和模块默认创建在当前 GPU。
        torch.set_default_device("cuda")
        # 根据配置在 GPU 上构建当前 rank 的模型分片。
        self.model = Qwen3ForCausalLM(hf_config)
        # 从模型路径加载并分发当前 rank 所需的权重。
        load_model(self.model, config.model)
        # 创建 logits 采样器；实际采样仅由 rank 0 完成。
        self.sampler = Sampler()
        # 用代表性批次预热内核，并测量峰值显存。
        self.warmup_model()
        # 根据预热后的可用显存分配固定大小的 KV Cache。
        self.allocate_kv_cache()
        # 未强制 eager 时，为 Decode 批次预先捕获 CUDA Graph。
        if not self.enforce_eager:
            # 捕获多种批大小的静态图以降低 Decode 调度开销。
            self.capture_cudagraph()
        # 恢复默认设备，避免影响此后在主机上创建的数据。
        torch.set_default_device("cpu")
        # 恢复调用前的默认浮点类型。
        torch.set_default_dtype(default_dtype)

        # 仅在多卡 Tensor Parallel 模式下建立主从命令通道。
        if self.world_size > 1:
            # rank 0 作为控制进程，负责创建共享内存。
            if rank == 0:
                # 创建 1 MiB 固定共享缓冲区传递方法名和参数。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 等待共享内存创建完成后再放行其他 rank。
                dist.barrier()
            # 非零 rank 作为工作进程，等待并执行 rank 0 的命令。
            else:
                # 确保 rank 0 已完成共享内存创建。
                dist.barrier()
                # 按名称附着到 rank 0 创建的共享内存。
                self.shm = SharedMemory(name="nanovllm")
                # 进入阻塞式命令循环，不再从构造函数返回。
                self.loop()

    # 释放共享资源、CUDA Graph 与分布式进程组。
    def exit(self):
        # 单卡模式没有共享内存，因此无需执行本段清理。
        if self.world_size > 1:
            # 关闭当前进程持有的共享内存句柄。
            self.shm.close()
            # 等待所有 rank 都不再访问共享内存。
            dist.barrier()
            # 仅创建者应删除共享内存对象。
            if self.rank == 0:
                # 从系统命名空间移除共享内存段。
                self.shm.unlink()
        # 只有捕获过 CUDA Graph 才存在这些属性。
        if not self.enforce_eager:
            # 显式释放 CUDA Graph 与其内存池引用。
            del self.graphs, self.graph_pool
        # 等待该 GPU 上已提交的工作完成，避免销毁时仍有异步任务。
        torch.cuda.synchronize()
        # 销毁 NCCL 通信组并释放其资源。
        dist.destroy_process_group()

    # 供非零 rank 长期运行：读取 rank 0 命令并在本地执行相同方法。
    def loop(self):
        # 持续等待控制进程下发下一条命令。
        while True:
            # 从共享内存读取方法名和参数列表。
            method_name, args = self.read_shm()
            # 在当前 rank 调用同名方法；非零 rank 不会再次广播。
            self.call(method_name, *args)
            # 收到退出命令后结束工作进程循环。
            if method_name == "exit":
                # 离开循环，使从进程可以正常退出。
                break

    # 由非零 rank 从共享内存读取一条由 rank 0 发布的命令。
    def read_shm(self):
        # 防止错误地在单卡或控制 rank 上调用。
        assert self.world_size > 1 and self.rank > 0
        # 阻塞至 rank 0 设置本 rank 的 Event。
        self.event.wait()
        # 读取缓冲区前 4 字节记录的有效负载长度。
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 反序列化紧随长度字段的方法名和参数。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清除通知，准备接收下一条命令。
        self.event.clear()
        # 把命令名称和参数分开交给调用循环。
        return method_name, args

    # 由 rank 0 将一条调用命令广播到所有 Tensor Parallel 从进程。
    def write_shm(self, method_name, *args):
        # 只允许多卡模式的控制 rank 写共享内存。
        assert self.world_size > 1 and self.rank == 0
        # 将方法名和可序列化参数打包为字节负载。
        data = pickle.dumps([method_name, *args])
        # 计算负载字节数，以便接收方确定读取边界。
        n = len(data)
        # 用固定 4 字节小端整数写入长度前缀。
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 将序列化负载写入共享内存的长度字段之后。
        self.shm.buf[4:n+4] = data
        # 逐个遍历所有非零 rank 对应的唤醒事件。
        for event in self.event:
            # 通知一个从进程读取并执行这条命令。
            event.set()

    # 在本 rank 调用命令；rank 0 会先让其他 rank 同步执行。
    def call(self, method_name, *args):
        # 控制 rank 需要先将调用复制给全部从进程。
        if self.world_size > 1 and self.rank == 0:
            # 发布方法名与参数到共享内存并唤醒工作进程。
            self.write_shm(method_name, *args)
        # 按字符串查找实例方法，使共享内存可传递调用意图。
        method = getattr(self, method_name, None)
        # 执行目标方法，并将其返回值交还本地调用者。
        return method(*args)

    # 运行一个最大代表性 Prefill 批次以初始化内核并测出临时峰值显存。
    def warmup_model(self):
        # 归还 PyTorch 缓存分配器中可释放的空闲显存。
        torch.cuda.empty_cache()
        # 从零开始统计本次预热期间的峰值张量显存。
        torch.cuda.reset_peak_memory_stats()
        # 读取批 Token 上限和单序列长度上限。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 选择不超过两项上限的预热序列长度。
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 计算在 Token 和序列数约束下可并行的预热序列数。
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 构造由占位 Token 组成的代表性请求序列。
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        # 遍历每个预热序列以标记完整提示词应被执行。
        for seq in seqs:
            # 将整条占位序列设为当前轮的 Prefill 输入。
            seq.num_scheduled_tokens = seq_len
        # 执行一次真实 Prefill，触发内核编译与临时工作区分配。
        self.run(seqs, True)
        # 清掉预热留下的可复用临时缓存，为 KV Cache 腾出显存。
        torch.cuda.empty_cache()

    # 按目标显存利用率计算容量，并把预分配 KV Cache 注入所有注意力层。
    def allocate_kv_cache(self):
        # 取出配置的局部引用，缩短后续访问路径。
        config = self.config
        # 取出模型结构参数和权重数据类型。
        hf_config = config.hf_config
        # 查询当前 GPU 的空闲与总显存字节数。
        free, total = torch.cuda.mem_get_info()
        # 计算模型权重和保留分配当前占据的显存。
        used = total - free
        # 获取预热过程中的 PyTorch 张量峰值分配。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 获取当前仍被 PyTorch 张量占用的显存。
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 将 KV 头按 Tensor Parallel rank 切分。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 优先使用显式 head_dim，否则由隐藏维和注意力头数推导。
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算一块跨全部层、同时含 Key 和 Value 的 KV Cache 字节数。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 在显存利用率预算内扣除模型和预热峰值后，换算可容纳的 Block 数。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        # 若无法容纳任何 Block，配置或 GPU 显存不足以运行推理。
        assert config.num_kvcache_blocks > 0
        # 在默认 CUDA 设备和模型精度下分配连续的 Key/Value 缓存张量。
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 跟踪当前遍历到第几个注意力层的 KV Cache 切片。
        layer_id = 0
        # 深度遍历模型的全部子模块。
        for module in self.model.modules():
            # 仅识别约定拥有 Key/Value Cache 属性的注意力模块。
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 为该层绑定其专属的 Key Cache 视图。
                module.k_cache = self.kv_cache[0, layer_id]
                # 为该层绑定其专属的 Value Cache 视图。
                module.v_cache = self.kv_cache[1, layer_id]
                # 移动到下一层对应的 KV Cache 切片。
                layer_id += 1

    # 将各序列长度不同的 Block Table 补齐并传输到 GPU。
    def prepare_block_tables(self, seqs: list[Sequence]):
        # 找出批次中最长的逻辑 Block Table 长度。
        max_len = max(len(seq.block_table) for seq in seqs)
        # 用 -1 将较短映射补齐为规则二维数组。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 先置于页锁定主机内存，再异步复制为 GPU int32 张量。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 返回供注意力内核按逻辑 Block 查询物理地址的二维表。
        return block_tables

    # 为批量 Prefill 构造拼接 Token、位置、FlashAttention 长度信息和 KV 写入槽位。
    def prepare_prefill(self, seqs: list[Sequence]):
        # 收集所有序列本轮待计算 Token，随后拼接为一维输入。
        input_ids = []
        # 收集每个待计算 Token 在各自序列中的绝对位置。
        positions = []
        # 以 0 开头累计每条序列的 Query 长度，供变长注意力使用。
        cu_seqlens_q = [0]
        # 以 0 开头累计每条序列的 Key 长度，包含命中的前缀缓存。
        cu_seqlens_k = [0]
        # 跟踪批内最长 Query 序列，供注意力内核选择执行配置。
        max_seqlen_q = 0
        # 跟踪批内最长 Key 序列，供注意力内核选择执行配置。
        max_seqlen_k = 0
        # 收集每个新 Token 应写入 Paged KV Cache 的线性槽位。
        slot_mapping = []
        # 无前缀缓存时不需要向注意力层传递 Block Table。
        block_tables = None
        # 逐条处理本轮被调度的 Prefill 序列。
        for seq in seqs:
            # 跳过已由前缀缓存复用或此前分块 Prefill 完成的 Token。
            start = seq.num_cached_tokens
            # 当前轮实际送入模型计算的 Query Token 数。
            seqlen_q = seq.num_scheduled_tokens
            # 计算当前轮 Token 切片的右开边界。
            end = start + seqlen_q
            # 注意力可见的 Key 长度等于已缓存加当前计算 Token 的总长度。
            seqlen_k = end
            # 将这条序列本轮的 Token 追加到批量一维输入。
            input_ids.extend(seq[start:end])
            # 追加与 Token 对应的绝对位置编码索引。
            positions.extend(range(start, end))
            # 记录下一条 Query 在拼接张量中的起始偏移。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            # 记录下一条 Key 在逻辑上下文中的累计偏移。
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            # 更新批次内最长的新增 Query 长度。
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            # 更新批次内最长的完整 Key 上下文长度。
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # warmup  # 预热序列没有真实 KV Cache 映射，无需准备写入槽位。
            if not seq.block_table:
                # 保留前述普通注意力输入信息，跳过 Paged KV Cache 地址计算。
                continue
            # 定位本轮第一个 Token 所在的逻辑 Block。
            start_block = start // self.block_size
            # 以向上取整方式定位本轮末尾之后的逻辑 Block 边界。
            end_block = (end + self.block_size - 1) // self.block_size
            # 遍历本轮涉及的每一个逻辑 Block。
            for i in range(start_block, end_block):
                # 计算该物理 Block 在线性 KV Cache 中的首槽位。
                slot_start = seq.block_table[i] * self.block_size
                # 首个 Block 可能从中间位置开始写入。
                if i == start_block:
                    # 跳过此前已缓存或已计算的 Block 内 Token 槽位。
                    slot_start += start % self.block_size
                # 非最后一个 Block 必定被本轮 Token 填满。
                if i != end_block - 1:
                    # 取该物理 Block 的完整右开边界。
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                # 最后一个 Block 可能只写入其部分槽位。
                else:
                    # 以本轮结束位置计算最后一个有效槽位之后的边界。
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                # 依 Token 顺序添加连续 KV 写入槽位。
                slot_mapping.extend(range(slot_start, slot_end))
        # prefix cache  # Key 总长度较大说明至少有一段前缀已缓存。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            # 将逻辑到物理的 Block 映射上传给注意力内核读取历史 KV。
            block_tables = self.prepare_block_tables(seqs)
        # 将拼接 Token 异步传到 GPU。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # 将绝对位置索引异步传到 GPU。
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # 将 Query 累计长度转为 GPU int32 张量。
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 将 Key 累计长度转为 GPU int32 张量。
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 将 KV Cache 写入槽位异步传到 GPU。
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 注册 Prefill 注意力所需的批次上下文。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        # 返回模型前向直接需要的 Token 与位置张量。
        return input_ids, positions

    # 为批量 Decode 构造每条序列一个 Token 的输入和 Paged KV Cache 上下文。
    def prepare_decode(self, seqs: list[Sequence]):
        # 收集每条序列上轮末尾的 Token 作为当前 Decode 输入。
        input_ids = []
        # 收集这些 Token 对应的绝对位置。
        positions = []
        # 收集每个新 KV 条目要写入的物理槽位。
        slot_mapping = []
        # 收集每条序列当前可见的上下文长度。
        context_lens = []
        # 每条 Decode 序列本轮只计算一个 Token。
        for seq in seqs:
            # 将最后生成或提示词末尾 Token 加入模型输入。
            input_ids.append(seq.last_token)
            # 该输入 Token 的位置为当前序列长度减一。
            positions.append(len(seq) - 1)
            # 注意力应看到包括当前输入 Token 在内的完整上下文长度。
            context_lens.append(len(seq))
            # 计算当前 Token 在尾部物理 Block 内对应的 KV 写入槽位。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # 将 Decode Token 异步上传到 GPU。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # 将 Decode 位置异步上传到 GPU。
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # 将 KV 写入地址异步上传到 GPU。
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 将各序列上下文长度异步上传到 GPU。
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 为 Decode 始终准备 Paged KV Cache 的 Block 映射。
        block_tables = self.prepare_block_tables(seqs)
        # 注册 Decode 注意力所需的批次上下文。
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        # 返回模型前向直接需要的 Token 与位置张量。
        return input_ids, positions

    # 将每条序列的温度参数整理为 rank 0 采样器可用的 GPU 张量。
    def prepare_sample(self, seqs: list[Sequence]):
        # 按批次序列顺序提取温度值。
        temperatures = [seq.temperature for seq in seqs]
        # 以 float32 页锁定内存创建后异步复制到 GPU。
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        # 返回与 logits 批维一一对应的温度张量。
        return temperatures

    # 禁用梯度记录，执行一次模型前向并在 Decode 时优先复用 CUDA Graph。
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # Prefill、强制 eager 或超出图捕获上限的批次直接执行动态图前向。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 运行模型并将隐状态投影为词表 logits。
            return self.model.compute_logits(self.model(input_ids, positions))
        # 常规小批量 Decode 使用预捕获 CUDA Graph 减少 CPU 提交开销。
        else:
            # 取得真实 Decode 批大小。
            bs = input_ids.size(0)
            # 读取 prepare_decode 注册的动态注意力上下文。
            context = get_context()
            # 选择不小于真实批大小的最小预捕获图桶。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # 取得被 CUDA Graph 捕获的静态输入输出张量。
            graph_vars = self.graph_vars
            # 将真实 Token 覆盖到静态输入缓冲区的有效前缀。
            graph_vars["input_ids"][:bs] = input_ids
            # 将真实位置覆盖到静态位置缓冲区的有效前缀。
            graph_vars["positions"][:bs] = positions
            # 先将全部槽位设为无效，避免较小批次遗留旧地址。
            graph_vars["slot_mapping"].fill_(-1)
            # 写入真实批次每个 Token 的 KV Cache 写入槽位。
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # 清零全部上下文长度，避免较小批次遗留旧长度。
            graph_vars["context_lens"].zero_()
            # 写入真实批次每条序列的上下文长度。
            graph_vars["context_lens"][:bs] = context.context_lens
            # 覆盖有效批次和有效 Block 列的物理映射。
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 在更新后的静态缓冲区上重放已捕获的 GPU 操作图。
            graph.replay()
            # 从图输出的有效批次隐状态计算并返回 logits。
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 准备一个调度批次、执行各 rank 的前向，并由 rank 0 采样下一个 Token。
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 根据阶段构造对应的模型输入与注意力上下文。
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 仅控制 rank 准备采样参数，其他 rank 只参与并行前向。
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 执行模型前向得到当前批次的词表 logits。
        logits = self.run_model(input_ids, positions, is_prefill)
        # 仅 rank 0 将 logits 按温度采样并转为 Python Token ID 列表。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 清除线程/进程局部注意力上下文，避免下一批复用陈旧张量。
        reset_context()
        # rank 0 返回采样结果，其他 rank 返回 None。
        return token_ids

    # 禁用梯度记录并预先捕获多个 Decode 批大小的 CUDA Graph。
    @torch.inference_mode()
    def capture_cudagraph(self):
        # 取出推理配置的局部引用。
        config = self.config
        # 取出模型隐藏维度等结构参数。
        hf_config = config.hf_config
        # CUDA Graph 仅为最多 512 条 Decode 序列建立静态缓冲区。
        max_bs = min(self.config.max_num_seqs, 512)
        # 计算单条最长序列可能需要的最大逻辑 Block 数。
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 创建静态 Decode Token 输入缓冲区。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        # 创建静态位置输入缓冲区。
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # 创建静态 KV 写入槽位缓冲区。
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        # 创建静态上下文长度缓冲区。
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # 创建静态逻辑到物理 Block 映射缓冲区。
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # 创建静态模型隐藏状态输出缓冲区。
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 定义小批量指数桶和大批量每 16 对齐的图捕获批大小。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # 初始化“批大小桶 -> CUDA Graph”映射。
        self.graphs = {}
        # 延后保存首个图的内存池，以便所有图共享分配。
        self.graph_pool = None

        # 由大到小捕获，使后续图可以复用最大的内存池。
        for bs in reversed(self.graph_bs):
            # 为当前静态批大小创建一个空 CUDA Graph 对象。
            graph = torch.cuda.CUDAGraph()
            # 为图中的 Decode 注意力注册切片后的静态上下文。
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup  # 在图捕获前先运行一次，完成惰性初始化。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 在共享内存池上开始记录该批大小的 CUDA 操作。
            with torch.cuda.graph(graph, self.graph_pool):
                # capture  # 捕获模型前向，并把结果写入静态输出缓冲区。
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 第一个捕获的图需要提供供后续图共享的内存池。
            if self.graph_pool is None:
                # 保存 CUDA Graph 内存池句柄。
                self.graph_pool = graph.pool()
            # 按静态批大小登记可重放的图。
            self.graphs[bs] = graph
            # 等待捕获和预热工作完成，确保下一轮捕获状态稳定。
            torch.cuda.synchronize()
            # 清除本轮静态上下文，避免泄漏到下一个批大小。
            reset_context()

        # 保存所有被图捕获的静态张量，以便 Decode 前逐批更新。
        self.graph_vars = dict(
            # 保存静态 Token 输入缓冲区。
            input_ids=input_ids,
            # 保存静态位置输入缓冲区。
            positions=positions,
            # 保存静态 KV 写入槽位缓冲区。
            slot_mapping=slot_mapping,
            # 保存静态上下文长度缓冲区。
            context_lens=context_lens,
            # 保存静态 Block Table 缓冲区。
            block_tables=block_tables,
            # 保存静态隐藏状态输出缓冲区。
            outputs=outputs,
        # 完成供 run_model 更新并重放的静态变量字典。
        )
