import torch
from stable_baselines3 import PPO
import os

# 获取 Transformer 层的权重
def get_transformer_weights(model):
    transformer_params = {}
    for name, param in model.policy.features_extractor.transformer_encoder.transformer.named_parameters():
        transformer_params[name] = param.clone().detach()
    return transformer_params

# 计算权重变化
def compare_weights(init_weights, trained_weights):
    weight_diff = {}
    for name in init_weights:
        diff = (trained_weights[name] - init_weights[name]).norm().item()  # L2 范数
        mean_change = (trained_weights[name] - init_weights[name]).abs().mean().item()  # 平均绝对变化
        weight_diff[name] = (diff, mean_change)
    return weight_diff

# 1. 训练前 Transformer 权重

init_model_path = r'agent_models/info_check/model0'  # 你的初始模型
init_model = PPO.load(init_model_path)
init_weights = get_transformer_weights(init_model)

# 2. 训练后 Transformer 权重
trained_model_path = r'agent_models/info_check/model1'  # 你的训练 5M 步的模型
trained_model = PPO.load(trained_model_path)
trained_weights = get_transformer_weights(trained_model)

# 3. 计算 Transformer 权重的变化
weight_changes = compare_weights(init_weights, trained_weights)

# 4. 输出变化情况
for name, (diff, mean_change) in weight_changes.items():
    print(f"{name} | L2 Norm Change: {diff:.6f} | Mean Absolute Change: {mean_change:.6f}")
