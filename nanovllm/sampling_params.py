# 导入数据类装饰器，用于声明采样配置字段。
from dataclasses import dataclass


# 将采样配置声明为使用 slots 的轻量数据类。
@dataclass(slots=True)
# 定义一次文本生成请求所使用的采样控制参数。
class SamplingParams:
    # 控制 softmax 分布平滑程度的随机采样温度。
    temperature: float = 1.0
    # 限制该请求最多新增生成的 Token 数。
    max_tokens: int = 64
    # 指示生成时是否忽略模型输出的 EOS 终止符。
    ignore_eos: bool = False

    # 在创建采样配置后校验采样温度的有效性。
    def __post_init__(self):
        # 拒绝近似零温度，当前实现不支持贪心解码。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
