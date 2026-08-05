# 从零基础理解 Nano-vLLM：学习与阅读路线

> 目标：不是立刻看懂每一个 CUDA 或 PyTorch API，而是循序建立“用户请求如何变成 GPU 上的下一个 token”这条主线。读完后，你应能解释：为什么要区分 prefill/decode、KV Cache 如何分配与复用、调度器如何组批、模型层如何完成注意力计算、以及多卡如何协同。

## 先建立正确预期

Nano-vLLM 是一个轻量级、离线运行的 Qwen3 推理引擎，而不是 Web 服务。它会在本地加载模型权重，接收一批文本或 token ID，然后反复执行：

```text
prompt
  → tokenizer
  → Sequence（请求运行状态）
  → Scheduler（本轮该算谁）
  → ModelRunner（准备 GPU 输入）
  → Qwen3 模型 + Attention
  → 采样下一个 token
  → 更新 KV Cache / 判断结束
  → 返回文本
```

建议始终围绕两个问题阅读：

1. 这一段代码在“控制哪个请求、使用哪块显存、计算哪个 token”？
2. 这项设计解决了什么资源问题，又带来了什么复杂度？

## 第一部分：开始前需要的基础知识

不需要先学完所有知识再读代码。按下面顺序“学一点、读一点、卡住再回补”效率最高。

| 阶段 | 需要掌握的最小知识 | 在项目中对应什么 | 学到什么程度即可 |
|---|---|---|---|
| 1. Python | 类、继承、dataclass、枚举、列表/dict/deque、切片、装饰器、上下文与序列化 | 全仓库 | 能读懂 `class`、`@dataclass`、`@property`、`pickle` 即可 |
| 2. PyTorch | Tensor 形状、dtype、device、`nn.Module`、Parameter、`F.linear`、`inference_mode` | `models/`、`layers/` | 能跟踪张量形状和 GPU/CPU 位置 |
| 3. Transformer | token、embedding、Q/K/V、attention、MLP、残差、RMSNorm、RoPE、logits、softmax | `models/qwen3.py`、`layers/` | 能解释“输入 token 如何预测下一个 token” |
| 4. 自回归推理 | prefill、decode、EOS、temperature、采样、上下文长度 | `llm_engine.py`、`sampling_params.py` | 能解释生成为什么是一轮一个 token |
| 5. GPU 性能 | 显存、带宽、kernel、pinned memory、异步 H2D | `model_runner.py` | 知道优化目标是减少显存与调度开销 |
| 6. LLM Serving | continuous batching、KV Cache、paged attention、prefix cache、抢占 | `scheduler.py`、`block_manager.py` | 能解释本项目最核心的系统设计 |
| 7. 多卡与内核（进阶） | NCCL collective、Tensor Parallel、FlashAttention、Triton、CUDA Graph、`torch.compile` | `linear.py`、`attention.py`、`model_runner.py` | 先会说明作用，不要求手写 kernel |

### 1. Python 必备知识清单

先看下面概念；不懂时可直接在文件注释旁查文档：

- `dataclass(slots=True)`：用较少样板代码定义配置/参数对象，并限制动态属性。
- `Enum`：把请求状态写成 `WAITING`、`RUNNING`、`FINISHED`，避免魔法字符串。
- `deque`：高效从两端入队/出队，适合调度队列。
- `@property`：把计算型属性伪装成字段，例如 `seq.num_blocks`。
- `@classmethod`：不依赖对象状态的工具方法，例如计算块 hash。
- `@torch.inference_mode()`：推理时关闭梯度，减少显存和计算。
- `pickle` 与 `__getstate__`：多进程传递对象时，控制要传哪些字段。

练习：先能用自己的话解释 `Sequence` 中的 `num_tokens`、`num_prompt_tokens`、`num_cached_tokens`、`num_scheduled_tokens` 四个数字各自表示什么。

### 2. Transformer 必备知识清单

这不是训练项目，所以先忽略反向传播和优化器。重点是一次前向推理：

```text
token id
  → Embedding
  → 重复 N 次：RMSNorm → Self-Attention → 残差 → RMSNorm → MLP → 残差
  → LM Head
  → logits（词表中每个 token 的分数）
  → sampling
  → 下一个 token id
```

需要理解的最小定义：

- **token**：文本被 tokenizer 切分后的整数 ID。
- **embedding**：把 token ID 查表为向量。
- **Q/K/V**：注意力机制中用于“查询、匹配、取值”的三组向量。
- **causal attention**：位置 *i* 只能看见它自己和之前的 token，不能看未来。
- **logits**：softmax 前的词表分数。
- **temperature**：调节分布随机性；越小越偏向高分 token。
- **EOS**：生成结束 token。
- **RoPE**：给 Q/K 注入位置信息的旋转位置编码。
- **GQA**：多个 Query head 共享较少的 Key/Value head，以节省 KV Cache。

练习：画出 prompt `"Hello"` 生成两个 token 时的序列：先处理所有 prompt token，再分别处理第一个和第二个生成 token。

### 3. 为什么 LLM 推理最关键的是 KV Cache

若每生成一个 token 都重新计算整段 prompt，复杂度会快速增长。模型在处理一个 token 时得到的 K/V 会写入 KV Cache；后续 decode 只计算最新 token 的 Q/K/V，再读取历史缓存。

```text
第 1 轮：处理 prompt 的所有 token，写入其 K/V
第 2 轮：处理第一个新 token，读取 prompt 的历史 K/V，写入新 K/V
第 3 轮：处理第二个新 token，读取更长历史 K/V，写入新 K/V
```

本项目进一步把 KV Cache 切成固定 block。这样多个请求的块可以非连续放置、按需增加、共享完整前缀；这就是 paged KV Cache 的动机。

### 4. 高性能概念的阅读优先级

先理解“为什么”，再理解 API 细节：

| 技术 | 要解决的问题 | 先记住的结论 |
|---|---|---|
| Continuous batching | 请求生成长度不同，静态 batch 会浪费 GPU | 每轮动态选择仍活跃的请求一起计算 |
| Chunked prefill | 长 prompt 会独占一个批次 | 将长 prompt 分段填充进预算 |
| Prefix cache | 相同系统提示词重复计算 | 复用已计算的完整前缀 KV block |
| FlashAttention | 注意力矩阵很大、显存访问昂贵 | 用专用 kernel 流式计算，避免显式大矩阵 |
| Triton | 通用 PyTorch 操作不够贴合数据布局 | 自定义 GPU kernel 将 K/V 写到指定 slot |
| CUDA Graph | decode 每步有较多 CPU/kernel launch 开销 | 捕获固定形状执行图并重复回放 |
| Tensor Parallel | 单卡放不下模型或算力不足 | 将一层权重和计算切到多张 GPU，再 collective 汇总 |

## 第二部分：推荐阅读顺序

下面的顺序刻意从“用户能看到的 API”走到“最底层的 GPU 实现”。不要一开始就读 `attention.py` 或 `model_runner.py` 的全部细节。

### 第 0 步：先跑通或模拟一次使用

依次阅读：

1. `README.md`
2. `example.py`
3. `bench.py`
4. `nanovllm/__init__.py`

先回答：用户怎样创建 `LLM`？输入是什么？输出是什么？benchmark 在测什么？

如果本机没有合适 NVIDIA GPU、Qwen3 权重或依赖，不必强行运行；可以只根据 `example.py` 跟踪调用关系。

### 第 1 步：读配置和采样参数

顺序：

1. `nanovllm/config.py`
2. `nanovllm/sampling_params.py`
3. `nanovllm/llm.py`

重点问题：

- `max_num_batched_tokens` 与 `max_num_seqs` 分别限制什么？
- 为什么 `max_model_len` 会受 Hugging Face 配置限制？
- `gpu_memory_utilization` 为什么不是 1？
- `temperature`、`max_tokens`、`ignore_eos` 如何影响一条请求？

读完后你应知道：模型与硬件相关配置在初始化阶段固定；每个请求只带很少的采样配置。

### 第 2 步：读请求状态 `Sequence`

文件：`nanovllm/engine/sequence.py`

这是最重要的起点。为一个具体例子写表：prompt 有 300 tokens，已生成 2 个 token，block size 为 256 时，各字段应该是多少。

| 字段 | 你要回答的问题 |
|---|---|
| `status` | 请求在等待、运行还是结束？ |
| `token_ids` | 包含 prompt 还是只包含输出？ |
| `num_prompt_tokens` | 如何切分 prompt / completion？ |
| `num_cached_tokens` | 已有多少 token 的 KV 已写入缓存？ |
| `num_scheduled_tokens` | 本轮 GPU 要处理多少 token？ |
| `block_table` | 逻辑 token block 映射到哪块物理 KV 显存？ |
| `is_prefill` | 当前序列需要走哪条计算路径？ |

### 第 3 步：读入口编排 `LLMEngine`

文件：`nanovllm/engine/llm_engine.py`

先只看四个方法：`__init__`、`add_request`、`step`、`generate`。

```text
generate
  → add_request：prompt 变成 Sequence 并入 waiting
  → while not is_finished
       → step
            → scheduler.schedule
            → model_runner.run
            → scheduler.postprocess
  → completion token IDs 解码为 text
```

重点：`LLMEngine` 是 Facade；它不自己决定调度策略，也不自己实现 Transformer 数学，只负责把各模块串起来。

### 第 4 步：读调度器 `Scheduler`

文件：`nanovllm/engine/scheduler.py`

建议分两遍：

1. 第一遍只看 `waiting`、`running`、`add`、`is_finished`，理解两条队列。
2. 第二遍逐行模拟 `schedule()`：先 prefill，无法安排 prefill 时才 decode；decode 没有新块时如何抢占。

要画出的状态转移：

```text
WAITING --prompt 全部 prefill--> RUNNING --EOS/max_tokens--> FINISHED
   ^                                  |
   └--------KV 不足，被抢占------------┘
```

关键追问：为什么 prefill 优先？为什么 chunk 只允许队首请求？为什么抢占后要重新 prefill，而不是把 KV 拷贝到 CPU？

### 第 5 步：读内存资源管理 `BlockManager`

文件：`nanovllm/engine/block_manager.py`

读法：先无视 xxHash，实现一个“块池”的心智模型：

```text
free_block_ids：可分配的物理块
used_block_ids：正在被请求使用的物理块
seq.block_table：这个请求使用的物理块顺序
ref_count：一个块被多少请求共享
```

再读前缀缓存：

1. `compute_hash` 如何把“前缀 hash + 当前完整 token block”变成链式身份？
2. `can_allocate` 如何判断可复用多少块、是否还有容量？
3. `allocate` 如何增加共享块引用计数？
4. `deallocate` 如何回收，但保留 hash/token 内容以供以后命中？
5. `hash_blocks` 为什么只在完整 block 计算完成后建立索引？

理解本文件后，项目最难的“显存生命周期”部分就基本过关了。

### 第 6 步：读 GPU 输入准备与执行器 `ModelRunner`

文件：`nanovllm/engine/model_runner.py`

该文件很长，严格按此顺序：

1. `__init__`：进程、GPU、模型、warmup、KV Cache、CUDA Graph 的初始化顺序。
2. `allocate_kv_cache`：先看张量形状和单块字节公式，不必立即推导每个维度。
3. `prepare_prefill`：把多个变长请求拼成一条 input tensor，并生成累计长度和 slot mapping。
4. `prepare_decode`：理解每个请求为什么只发送 `last_token`。
5. `run`：把准备、模型、采样和 Context 清理串起来。
6. `run_model` / `capture_cudagraph`：最后再理解 eager 与 CUDA Graph 双路径。
7. `read_shm` / `write_shm` / `loop`：最后理解 rank 0 如何通知其他 TP worker。

读到 `slot_mapping` 时，回看 `BlockManager`：它就是“当前逻辑 token 应写到物理 KV 大张量中的哪个位置”。

### 第 7 步：读运行上下文与注意力

顺序：

1. `nanovllm/utils/context.py`
2. `nanovllm/layers/attention.py`

`Context` 是一次 GPU 执行的元数据包。它用进程全局变量避免层层传递长参数列表。`Attention.forward` 的高层流程是：

```text
当前轮生成 q/k/v
  → Triton kernel 按 slot_mapping 写入 k/v cache
  → prefill：FlashAttention 计算变长因果注意力
  → decode：FlashAttention 从分页 cache 读取历史 K/V
  → 返回 attention 输出
```

此处不要先钻 `tl.load`、stride 和 pointer arithmetic。第一目标是知道输入/输出是什么、cache 为什么能被正确读写；第二轮才学习 Triton 语法。

### 第 8 步：读 Qwen3 结构

文件：`nanovllm/models/qwen3.py`

建议沿 `Qwen3ForCausalLM → Qwen3Model → Qwen3DecoderLayer → Qwen3Attention/Qwen3MLP` 向下读。每读一个类，都记录：输入张量形状、输出张量形状、是否改变 token 数、是否需要跨 GPU 通信。

```text
input_ids [token_count]
  → VocabParallelEmbedding [token_count, hidden_size]
  → N 个 DecoderLayer [token_count, hidden_size]
  → RMSNorm [token_count, hidden_size]
  → ParallelLMHead [batch_size, vocab_size]
```

注意：prefill 时 LM Head 只对每个序列最后一个 token 计算 logits；decode 时每个序列恰好只有一个 token，所以两者都只需要“一条请求一个 logits 行”。

### 第 9 步：读通用 layers

推荐顺序：

1. `layers/sampler.py`：从 logits 产生 token，最短也最直观。
2. `layers/activation.py`：SwiGLU 的 SiLU 和乘法。
3. `layers/layernorm.py`：RMSNorm 与残差融合。
4. `layers/rotary_embedding.py`：RoPE 缓存与旋转。
5. `layers/linear.py`：Tensor Parallel 最关键的分片逻辑。
6. `layers/embed_head.py`：词表并行 embedding 和输出 head。

对于 `linear.py`，记住三句话即可：

- Column Parallel：按输出维切分，各卡可先独立计算。
- Row Parallel：按输入维切分，局部结果需要 `all_reduce` 相加。
- Vocab Parallel：词表按行切分，embedding 结果需要汇总，最终 logits 在 rank 0 聚合采样。

### 第 10 步：最后读权重加载

文件：`nanovllm/utils/loader.py`

这个文件解释为什么 Qwen3 中有 `packed_modules_mapping`：HF checkpoint 的 `q_proj/k_proj/v_proj` 是分开的，而运行时为了少 kernel/更方便 TP，代码把它们装进合并的 `qkv_proj`。同理，gate/up 被合并为一个投影。

读完时要能回答：权重加载不仅是 `copy_`，还要根据参数类型和 rank 把权重切到正确 GPU shard。

## 第三部分：一条请求的手工追踪练习

以两个请求为例：

```text
A：300 token prompt，最多生成 2 token
B：600 token prompt，最多生成 2 token
block_size = 256
```

请在纸上或表格中跟踪每轮：

| 轮次 | scheduler 选择 | 类型 | `num_scheduled_tokens` | block table 变化 | 输出 token 是否真正 append |
|---|---|---|---:|---|---|
| 1 | A / B 或 A | prefill | 受 batch token 预算限制 | 为 prompt 分配或复用块 | 仅 prompt 全部处理后才 append |
| 2 | 未完成 prompt | prefill | 剩余 chunk | 可能无新增块 | 同上 |
| 3 | 已完成 prompt 的序列 | decode | 每序列 1 | 跨块边界才追加物理块 | append 第一个 completion |
| 4 | 仍未结束序列 | decode | 每序列 1 | 继续写 KV | append 第二个 completion 或结束 |

能正确完成这张表，就说明你已经真正理解了调度、KV Cache 和自回归生成的衔接。

## 第四部分：建议的学习节奏

### 第一天：建立大图（2–3 小时）

- 阅读 README、example、config、sampling params、Sequence、LLMEngine。
- 画出 `generate → step → schedule → run → postprocess`。
- 不理解模型公式也没关系。

### 第二天：理解调度与缓存（3–4 小时）

- 读 Scheduler 与 BlockManager。
- 手工模拟 2 个不同长度请求。
- 重点理解：prefix cache、ref count、preemption、chunked prefill。

### 第三天：理解 GPU 数据面（3–4 小时）

- 读 ModelRunner 的 prepare/run 部分和 Context。
- 看 Attention 高层流程。
- 补学习 pinned memory、slot mapping、FlashAttention。

### 第四天：理解模型与多卡（3–5 小时）

- 读 Qwen3、RMSNorm、RoPE、Linear、Embedding/LM Head。
- 重点理解 Column/Row/Vocab Parallel 的通信位置。

### 第五天：复盘与改造（2–3 小时）

- 阅读 `docs/markdown/项目深度解析.md` 的性能、技术债和扩展性章节。
- 尝试回答：如果 QPS 增加 10 倍，最先改哪里？
- 自己写一个小图或伪代码解释单个 decode step。

## 最容易卡住的地方，以及正确解法

| 卡点 | 不要做什么 | 应该怎么做 |
|---|---|---|
| 看不懂 Tensor shape | 直接跳到 CUDA kernel | 先在纸上写每个张量的 batch/token/hidden/head 维度 |
| 看不懂 `slot_mapping` | 把它当抽象魔法 | 回到 block table：它只是逻辑 token 到物理 KV 槽位的映射 |
| 看不懂 `all_reduce` | 记 API 名称 | 画出每张卡算“部分和”，再相加得到完整输出 |
| 看不懂 CUDA Graph | 先研究 capture API | 先理解 decode 的 batch shape 为什么相对稳定 |
| 看不懂 FlashAttention | 阅读 kernel 源码 | 先知道它替代了普通 attention 的哪部分显存访问 |
| 看不懂权重 loader | 逐 tensor 调试 | 先理解 QKV/gate-up 为什么在运行时合并 |

## 最终掌握标准

当你可以不看代码回答以下问题，就算完成第一轮学习：

1. `LLM.generate()` 如何把字符串变成输出文本？
2. prefill 和 decode 的输入、计算特征、调度方式有什么差别？
3. 为什么 KV Cache 是推理系统的核心显存资源？
4. `Sequence.block_table` 和 `slot_mapping` 各解决什么问题？
5. prefix cache 如何避免不同上下文的错误复用？
6. 显存不足为什么会触发 preemption + recompute？
7. 为什么 CUDA Graph 主要用在 decode？
8. Column Parallel、Row Parallel、Vocab Parallel 分别在哪里通信？
9. 本项目为什么是推理内核而不是完整线上服务？
10. 如果从零做生产服务，还缺哪些能力？

如果其中任一题答不清，回到对应文件的中文行级注释，再沿本文给出的前置概念补课；不要靠死记术语跳过因果关系。
