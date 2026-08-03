# 导入实现调度与推理流程的底层 LLMEngine 基类。
from nanovllm.engine.llm_engine import LLMEngine


# 以更简洁的公共名称暴露完整的 LLMEngine 功能。
class LLM(LLMEngine):
    # 不新增实现，直接继承基类的全部接口与行为。
    pass
