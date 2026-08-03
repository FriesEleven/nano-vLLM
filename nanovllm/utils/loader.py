# 导入路径拼接工具，用于定位模型权重文件。
import os
# 导入通配符查找函数，用于枚举 safetensors 分片。
from glob import glob
# 导入 PyTorch，以标注加载的权重张量类型。
import torch
# 导入神经网络模块命名空间，以标注模型和参数类型。
from torch import nn
# 导入 safetensors 的安全只读打开接口。
from safetensors import safe_open


# 定义普通未分片参数的默认权重写入逻辑。
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # 将加载到 CPU 的权重原地复制到目标 Parameter 的底层数据中。
    param.data.copy_(loaded_weight)


# 定义从目录中读取 safetensors 权重并写入模型参数的函数。
def load_model(model: nn.Module, path: str):
    # 读取模型声明的打包模块映射；普通模型缺失该属性时使用空映射。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 遍历模型目录内所有可能的 safetensors 权重分片文件。
    for file in glob(os.path.join(path, "*.safetensors")):
        # 以 PyTorch 张量格式在 CPU 上安全地打开当前权重分片。
        with safe_open(file, "pt", "cpu") as f:
            # 逐个处理该分片中记录的权重名称。
            for weight_name in f.keys():
                # 检查权重名是否属于需要拆分加载的打包模块。
                for k in packed_modules_mapping:
                    # 当权重名称含有当前打包模块键时进入分片加载路径。
                    if k in weight_name:
                        # 取出目标参数名替换规则和该权重对应的分片编号。
                        v, shard_id = packed_modules_mapping[k]
                        # 将检查到的打包模块键替换为模型真实参数名。
                        param_name = weight_name.replace(k, v)
                        # 根据替换后的名称取得模型中待写入的参数对象。
                        param = model.get_parameter(param_name)
                        # 获取该参数专用的分片权重加载函数。
                        weight_loader = getattr(param, "weight_loader")
                        # 读取当前权重张量并带分片编号写入目标参数。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        # 已找到匹配规则，停止继续检查其他打包模块键。
                        break
                # 若内层 for 未因 break 退出，说明这是普通未打包权重。
                else:
                    # 直接按原权重名称取得模型中的对应参数。
                    param = model.get_parameter(weight_name)
                    # 优先使用参数自定义加载器，否则回退到直接复制的默认实现。
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 读取普通权重张量并写入对应参数。
                    weight_loader(param, f.get_tensor(weight_name))
