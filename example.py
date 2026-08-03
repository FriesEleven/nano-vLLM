# 导入操作系统工具，用于展开用户目录中的模型路径。
import os
# 导入本项目暴露的推理引擎和采样参数类型。
from nanovllm import LLM, SamplingParams
# 导入 Hugging Face 的自动分词器加载接口。
from transformers import AutoTokenizer


# 定义示例程序的主执行函数。
def main():
    # 将波浪号开头的本地 Qwen 模型目录展开为绝对路径。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 从模型目录加载与模型匹配的分词器及聊天模板。
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 创建单卡、即时执行模式的语言模型推理引擎。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 设置随机采样温度和每个请求最多生成的 Token 数。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # 准备尚未套用聊天模板的原始用户问题列表。
    prompts = [
        # 第一个待生成的英文自我介绍请求。
        "introduce yourself",
        # 第二个要求列出 100 以内质数的请求。
        "list all prime numbers within 100",
    ]
    # 将每条原始问题转换为该模型需要的聊天格式提示词。
    prompts = [
        # 按模型 tokenizer 的规则生成单条聊天提示词文本。
        tokenizer.apply_chat_template(
            # 构造只有 user 角色和当前问题内容的一轮对话。
            [{"role": "user", "content": prompt}],
            # 要求函数返回字符串而非已编码的 Token ID。
            tokenize=False,
            # 在末尾追加 assistant 开始生成所需的提示标记。
            add_generation_prompt=True,
        )
        # 遍历上一轮定义的每条原始问题。
        for prompt in prompts
    ]
    # 批量执行推理，并取得与提示词一一对应的生成结果。
    outputs = llm.generate(prompts, sampling_params)

    # 同步遍历原始聊天提示词和对应的模型输出。
    for prompt, output in zip(prompts, outputs):
        # 在每条结果前输出空行，改善终端可读性。
        print("\n")
        # 以可见转义形式打印当前提示词，便于检查格式。
        print(f"Prompt: {prompt!r}")
        # 从结果字典取出生成文本并以可见转义形式打印。
        print(f"Completion: {output['text']!r}")


# 仅在该文件被直接运行时启动示例主函数。
if __name__ == "__main__":
    # 执行上方定义的推理演示流程。
    main()
