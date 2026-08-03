# 注册解释器退出时的资源清理回调。
import atexit
# 枚举 Config 数据类字段，筛选可接受的初始化参数。
from dataclasses import fields
# 使用高精度单调时钟统计每一步的吞吐。
from time import perf_counter
# 创建可在终端或 Notebook 自适应显示的进度条。
from tqdm.auto import tqdm
# 从模型目录加载与权重匹配的 Hugging Face 分词器。
from transformers import AutoTokenizer
# 使用 PyTorch 推荐的多进程上下文启动 Tensor Parallel worker。
import torch.multiprocessing as mp

# 导入推理引擎的统一运行配置。
from nanovllm.config import Config
# 导入每个请求可独立指定的采样配置。
from nanovllm.sampling_params import SamplingParams
# 导入保存请求状态、Token 与 Block Table 的序列类型。
from nanovllm.engine.sequence import Sequence
# 导入负责 Continuous Batching 的调度器。
from nanovllm.engine.scheduler import Scheduler
# 导入每个 GPU rank 上执行模型的运行器。
from nanovllm.engine.model_runner import ModelRunner


# 对外暴露请求提交、批量生成和 Tensor Parallel 生命周期管理的高层引擎。
class LLMEngine:

    # 根据模型标识及可选配置创建分词器、调度器和各 GPU 的 ModelRunner。
    def __init__(self, model, **kwargs):
        # 收集 Config 声明的字段名，防止无关参数传入配置构造器。
        config_fields = {field.name for field in fields(Config)}
        # 仅保留名称匹配的可选配置参数。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 用模型路径/名称和过滤后的参数创建运行配置。
        config = Config(model, **config_kwargs)
        # 将全局逻辑序列 Block 大小与实际 KV Cache 配置保持一致。
        Sequence.block_size = config.kvcache_block_size
        # 保存非零 Tensor Parallel rank 的子进程对象，供退出时 join。
        self.ps = []
        # 保存用于唤醒每个子进程的 Event，交给 rank 0 ModelRunner 广播命令。
        self.events = []
        # 选择干净的 spawn 启动方式，避免 CUDA fork 带来的不安全状态。
        ctx = mp.get_context("spawn")
        # 为除 rank 0 以外的每张 Tensor Parallel GPU 创建 worker。
        for i in range(1, config.tensor_parallel_size):
            # 创建该 worker 专属的进程间命令通知事件。
            event = ctx.Event()
            # 配置子进程直接构造并运行对应 rank 的 ModelRunner。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # 启动 worker，使其初始化模型后进入共享内存命令循环。
            process.start()
            # 保存子进程句柄，以便关闭时等待其结束。
            self.ps.append(process)
            # 保存事件，供 rank 0 向该 worker 广播调用。
            self.events.append(event)
        # 在当前进程初始化控制 rank 的 ModelRunner。
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载快速分词器，将文本提示词转换为 Token ID。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 将分词器 EOS ID 写入配置，供调度器判断生成结束。
        config.eos = self.tokenizer.eos_token_id
        # 创建拥有 BlockManager 和等待/运行队列的调度器。
        self.scheduler = Scheduler(config)
        # 注册退出钩子，尽力回收 GPU、NCCL 和 worker 资源。
        atexit.register(self.exit)

    # 终止所有 rank 的运行器并等待子进程完全退出。
    def exit(self):
        # 由 rank 0 广播退出命令，同时清理自身 ModelRunner。
        self.model_runner.call("exit")
        # 删除主进程持有的运行器引用，释放其剩余 Python 对象。
        del self.model_runner
        # 逐个等待此前启动的 Tensor Parallel worker。
        for p in self.ps:
            # 阻塞至对应子进程完成 exit 命令并终止。
            p.join()

    # 将文本或预分词 Token 提示词封装为序列并加入调度队列。
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 文本提示词需要先经模型对应分词器编码。
        if isinstance(prompt, str):
            # 将文本转换为 Token ID 列表。
            prompt = self.tokenizer.encode(prompt)
        # 创建携带调度状态和采样参数的新请求序列。
        seq = Sequence(prompt, sampling_params)
        # 将新序列放入等待队列，等待下次调度。
        self.scheduler.add(seq)

    # 执行一次调度、模型前向、采样和状态更新，并返回本轮完成的请求。
    def step(self):
        # 选择本轮可执行的序列并确定处于 Prefill 还是 Decode 阶段。
        seqs, is_prefill = self.scheduler.schedule()
        # 以正数统计 Prefill Token、以负数编码 Decode 序列数用于吞吐显示。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 在全部 Tensor Parallel rank 执行前向，并由 rank 0 返回采样 Token。
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 提交 KV Cache、追加生成 Token，并处理完成或继续运行的状态。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 收集恰在本轮完成的序列 ID 及其补全 Token。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 返回完成结果与用于性能统计的本轮工作量。
        return outputs, num_tokens

    # 判断等待队列和运行队列是否都已清空。
    def is_finished(self):
        # 委托调度器根据内部队列状态给出答案。
        return self.scheduler.is_finished()

    # 同步生成一批请求的补全文本，并在可选进度条中展示实时吞吐。
    def generate(
        # 绑定引擎实例以访问分词器、调度器和模型运行器。
        self,
        # 接受文本提示词列表或预分词 Token ID 列表。
        prompts: list[str] | list[list[int]],
        # 接受全批次共享的配置，或与提示词一一对应的配置列表。
        sampling_params: SamplingParams | list[SamplingParams],
        # 控制是否显示生成进度条。
        use_tqdm: bool = True,
    # 返回按输入顺序排列的文本与 Token ID 结果字典列表。
    ) -> list[str]:
        # 创建以请求数量为总量的自适应宽度进度条。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 共享单个采样配置时，需要扩展成逐请求配置。
        if not isinstance(sampling_params, list):
            # 为每条提示词复用同一个 SamplingParams 对象。
            sampling_params = [sampling_params] * len(prompts)
        # 按顺序配对每条提示词及其采样配置。
        for prompt, sp in zip(prompts, sampling_params):
            # 将该请求转换为 Sequence 后送入等待队列。
            self.add_request(prompt, sp)
        # 以序列 ID 为键暂存完成结果，因为完成顺序可能不同于提交顺序。
        outputs = {}
        # 初始化两个阶段的最近一次吞吐数值。
        prefill_throughput = decode_throughput = 0.
        # 持续运行调度步，直到所有请求均完成。
        while not self.is_finished():
            # 记录本轮开始时间，用于计算吞吐。
            t = perf_counter()
            # 执行一轮端到端推理并获取本轮完成的结果。
            output, num_tokens = self.step()
            # 正值表示该轮执行了 Prefill。
            if num_tokens > 0:
                # 用本轮处理 Token 数除以耗时计算 Prefill 吞吐。
                prefill_throughput = num_tokens / (perf_counter() - t)
            # 非正值表示 Decode；其中负号编码了本轮解码序列数量。
            else:
                # 取反后用解码 Token 数除以耗时计算 Decode 吞吐。
                decode_throughput = -num_tokens / (perf_counter() - t)
            # 更新进度条末尾显示的阶段性性能指标。
            pbar.set_postfix({
                # 将 Prefill 吞吐格式化为整数 Token/s。
                "Prefill": f"{int(prefill_throughput)}tok/s",
                # 将 Decode 吞吐格式化为整数 Token/s。
                "Decode": f"{int(decode_throughput)}tok/s",
            # 提交本轮进度条附加信息。
            })
            # 处理这轮刚完成的每条序列。
            for seq_id, token_ids in output:
                # 按唯一序列 ID 保存其补全 Token。
                outputs[seq_id] = token_ids
                # 将已完成请求数增加一。
                pbar.update(1)
        # 关闭进度条并释放其终端/Notebook 显示资源。
        pbar.close()
        # 按递增序列 ID 恢复与输入提示词一致的顺序。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码每条补全 Token，并同时保留原始 Token ID。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        # 将最终结果列表返回给调用方。
        return outputs
