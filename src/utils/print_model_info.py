import torch
import os
from stable_baselines3 import PPO
from models.aux_obj_ppo import AuxObjPPO

MODEL_DIR = r'agent_models/info_check'
MODEL_NAME = 'info_check'
# MODEL_NAME = 'spmv_reward'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 1. 加载模型
model = PPO.load(MODEL_PATH)

# 为了统计模型中各个模块的参数数量，先写一个小工具函数
def count_parameters(module: torch.nn.Module) -> int:
    """
    返回一个torch模块（module）中所有参数的数量（numel）。
    """
    return sum(p.numel() for p in module.parameters())

# 2. 输出模型结构
# model.policy 包含了整个策略（含特征提取器、MLP、action_net、value_net等）
print("=== Model Policy Structure ===")
print(model.policy)

# 2.1 特征提取器（feature_extractor）
print("\n=== Feature Extractor Structure ===")
print(model.policy.features_extractor)

# 2.2 MLP（多层感知器）提取器，通常包含共享层、policy/action层和value层的中间部分
#     在默认的MlpPolicy中，mlp_extractor 会输出policy_latent和value_latent
print("\n=== MLP Extractor (shared layers) ===")
print(model.policy.mlp_extractor)

# 2.3 策略网络 (action_net)
print("\n=== Action (Policy) Network ===")
print(model.policy.action_net)

# 2.4 价值网络 (value_net)
print("\n=== Value Network ===")
print(model.policy.value_net)

# 3. 统计并输出模型总参数数量
total_params = count_parameters(model.policy)
print(f"\nTotal number of parameters in the policy: {total_params}")

# 4. 分别输出各子模块的参数数量
feature_extractor_params = count_parameters(model.policy.features_extractor)
mlp_extractor_params = count_parameters(model.policy.mlp_extractor)
action_net_params = count_parameters(model.policy.action_net)
value_net_params = count_parameters(model.policy.value_net)

print("=== Parameter counts by sub-module ===")
print(f"Feature Extractor: {feature_extractor_params}")
print(f"MLP Extractor:     {mlp_extractor_params}")
print(f"Action Network:    {action_net_params}")
print(f"Value Network:     {value_net_params}")
