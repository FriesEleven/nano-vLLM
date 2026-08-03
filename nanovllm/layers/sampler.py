# 提供 softmax、随机指数噪声和张量算子。
import torch
# 引入采样器继承的模块基类。
from torch import nn


# 定义从语言模型 logits 按温度进行随机采样的模块。
class Sampler(nn.Module):

    # 编译采样热点路径，减少每一步 Decode 的 Python 开销。
    @torch.compile
    # 对每条请求的 logits 施加温度，并使用 Gumbel-Max 等价形式抽取 token。
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 转为 float32，并按每条请求的温度沿词表维度缩放分数。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 将温度缩放后的 logits 归一化为词表概率分布。
        probs = torch.softmax(logits, dim=-1)
        # 通过指数噪声除法实现等价的 Gumbel-Max 随机采样，并取最大项的 token id。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # 返回每条请求采样出的下一个 token id。
        return sample_tokens
