# 导入操作系统工具，用于展开本地模型路径。
import os
# 导入计时模块，用于计算端到端吞吐量。
import time
# 导入随机整数生成器和随机种子设置函数。
from random import randint, seed
# 导入本项目的推理引擎和采样参数类型。
from nanovllm import LLM, SamplingParams
# 如需对照官方 vLLM，可改用下面这一行导入。
# from vllm import LLM, SamplingParams


# 定义离线批量推理基准测试的主函数。
def main():
    # 固定随机数序列，保证多次基准测试的输入可复现。
    seed(0)
    # 指定一次批量测试中要并发生成的序列数量。
    num_seqs = 256
    # 指定随机生成提示词的最大长度。
    max_input_len = 1024
    # 指定随机生成目标输出的最大长度；变量名保留原拼写。
    max_ouput_len = 1024

    # 将本地 Qwen 模型目录展开为绝对路径。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 创建启用 CUDA Graph 等优化的推理引擎，并限制最大上下文长度。
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 为每个请求随机生成 100 到最大输入长度之间的 Token ID 序列。
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # 为每个请求随机生成采样配置和 100 到最大输出长度之间的生成上限。
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # 下方注释说明切换到 vLLM 接口时所需的输入格式差异。
    # uncomment the following line for vllm
    # 将每条 Token ID 序列包装成 vLLM 所需的字典格式。
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 先执行一次极小请求，以完成可能影响测量的懒初始化和预热。
    llm.generate(["Benchmark: "], SamplingParams())
    # 记录正式批量推理开始的墙钟时间。
    t = time.time()
    # 关闭进度条后运行全部随机请求，避免终端输出干扰基准结果。
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    # 用结束时间减开始时间得到此次批处理的耗时秒数。
    t = (time.time() - t)
    # 汇总全部请求预期生成的最大 Token 数作为吞吐统计分子。
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # 计算每秒可生成的 Token 数。
    throughput = total_tokens / t
    # 格式化输出总 Token、耗时与吞吐量。
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


# 仅在直接运行该脚本时执行基准测试。
if __name__ == "__main__":
    # 启动上方定义的基准测试流程。
    main()
