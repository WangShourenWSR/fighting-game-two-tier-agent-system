from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import warnings
from gym import spaces
import gym
from numpy import ndarray
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space, preprocess_obs
from stable_baselines3.common.type_aliases import TensorDict

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torchvision import models
import numpy as np
    
class CustomCNN(BaseFeaturesExtractor):
    """
    : param observation_space: (gym.Space)
    : param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        additional_features_dim: List[int] = [0, 0],
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # To make the features dimision of the input of features extractors match the input of mlp_extractor,
        # need to minus the addtional_features_dim from the features_dim
        features_dim -= sum(additional_features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(features_dim),
            nn.Linear(features_dim, features_dim), 
            nn.ReLU(),
            # nn.BatchNorm1d(features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet18(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        additional_features_dim: List[int] = [0, 0],
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # To make the features dimision of the input of features extractors match the input of mlp_extractor,
        # need to minus the addtional_features_dim from the features_dim
        features_dim -= sum(additional_features_dim)
        # We assume CxHxW images (channels first)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use CustomResNetCNN "
            "only with images not with {}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        ).format(observation_space)

        # Load a pre-trained ResNet model, remove its fully connected layers, and freeze the layers
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity() # Remove the fully connected layer

        # Adapt ResNet's convolutional part to match your input space and requirement
        # Note: You might need to adjust this depending on your input image size and requirements
        n_input_channels = observation_space.shape[0]
        self.cnn.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Example of extending the ResNet model with additional layers
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Define the fully connected layers to match your specific features_dim
        self.linear = nn.Sequential(
            nn.Linear(512, features_dim),  # Adjust the in_features to match ResNet output
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Extract features using the CNN (ResNet)
        cnn_features = self.cnn(observations)
        # Pass the CNN features through the fully connected layers
        return self.linear(cnn_features)

class CustomResNet50(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use CustomResNetCNN "
            "only with images not with {}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        ).format(observation_space)

        # Load a pre-trained ResNet model, remove its fully connected layers, and freeze the layers
        self.cnn = models.resnet50(pretrained=False)
        self.cnn.fc = nn.Identity() # Remove the fully connected layer

        # Adapt ResNet's convolutional part to match your input space and requirement
        # Note: You might need to adjust this depending on your input image size and requirements
        n_input_channels = observation_space.shape[0]
        self.cnn.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Example of extending the ResNet model with additional layers
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Define the fully connected layers to match your specific features_dim
        self.linear = nn.Sequential(
            nn.Linear(2048, features_dim),  # Adjust the in_features to match ResNet output
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Extract features using the CNN (ResNet)
        cnn_features = self.cnn(observations)
        # Pass the CNN features through the fully connected layers
        return self.linear(cnn_features)

class CustomCombinedExtractor_ResNet18(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 512,
        normalized_image: bool = False,
        **kwargs
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = CustomResNet18(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
    

class CustomCombinedExtractor_Transformer(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 512,
        transformer_d_model: int = 128,
        seq_len: int = 100,
        nhead: int = 8,
        num_layers: int = 4,
        normalized_image: bool = False,
        **kwargs
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        pooling_type = kwargs.get("pooling_type", "attn_pool")
        if pooling_type == "attn_pool":
            self.transformer_encoder = ActionTransformerEncoderAttnPool(
            action_dim=12,          
            seq_len=seq_len,
            d_model=transformer_d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        elif pooling_type == "mean_pool":
            self.transformer_encoder = ActionTransformerEncoder(
            action_dim=12,          
            seq_len=seq_len,
            d_model=transformer_d_model,
            nhead=nhead,
            num_layers=num_layers,
            mean_pool=True
        )
        elif pooling_type == "last_element":
            self.transformer_encoder = ActionTransformerEncoder(
            action_dim=12,          
            seq_len=seq_len,
            d_model=transformer_d_model,
            nhead=nhead,
            num_layers=num_layers,
            mean_pool=False
        )
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = CustomResNet18(
                    subspace,
                    features_dim=cnn_output_dim,
                    additional_features_dim=[0, 0],
                    normalized_image=normalized_image
                )
                total_concat_size += cnn_output_dim

            elif key == "action_sequence":
                extractors[key] = nn.Identity()
                total_concat_size += transformer_d_model

            else:
                extractors[key] = nn.Flatten()
                flat_size = get_flattened_obs_dim(subspace)
                total_concat_size += flat_size

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
        
            if key == "action_sequence":
                seq_feat = self.transformer_encoder(observations[key])
                encoded_tensor_list.append(seq_feat)
            else:
                encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)

class PositionalEncoding(nn.Module):
        """Learnable Positional Embedding"""
        def __init__(self, max_seq_len=100, d_model=128):
            super().__init__()
            self.pe = nn.Embedding(max_seq_len, d_model)

        def forward(self, x):
            """
            x: [batch_size, seq_len, d_model]
            return: [batch_size, seq_len, d_model]
            """
            bsz, seq_len, _ = x.shape
            positions = th.arange(0, seq_len, device=x.device).unsqueeze(0) 
            positions = positions.expand(bsz, seq_len)                       
            return x + self.pe(positions)

class ActionTransformerEncoder(nn.Module):
    """
    针对past_actions的Transformer编码器
    输入: [batch_size, seq_len, action_dim]
    输出: [batch_size, d_model] —— 一个向量表示整段序列
    """
    def __init__(self,
                action_dim=12,
                seq_len=100,
                d_model=128,
                nhead=8,
                num_layers=4,
                mean_pool=False
                ):
        super().__init__()

        self.action_embedding = nn.Linear(action_dim, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len=seq_len, d_model=d_model)
        self.mean_pool = mean_pool
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: [batch_size, seq_len, action_dim]
        """
        x = self.action_embedding(x)    # -> [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)       
        x = self.transformer(x)         # -> [batch_size, seq_len, d_model]
        # Pooling
        # Use mean pooling if mean_pool is True, otherwise use the last hidden state
        if self.mean_pool:
            x = x.transpose(1, 2)           # [batch_size, d_model, seq_len]
            x = self.pool(x).squeeze(-1)    # [batch_size, d_model]
        else:
            x = x[:, -1, :]                 # [batch_size, d_model]
            # x = x[:, -3:, :].flatten(1)     # [batch_size, 3*d_model]

        return x

class ActionTransformerEncoderAttnPool(nn.Module):
    '''
    The regular pooling method may lose the information of the sequence, especially for triggering accurate actions.
    Here we use the attention pooling to extract the global feature of the sequence.
    '''
    def __init__(self, action_dim=12, seq_len=100, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.action_embedding = nn.Linear(action_dim, d_model)
        # Modifiy the max_seq_len to seq_len + 1 to include the CLS token
        self.pos_encoding = PositionalEncoding(max_seq_len=seq_len + 1, d_model=d_model)
        # The CLS token to be applied to the sequence head for global feature extraction
        self.cls_token = nn.Parameter(th.randn(1, 1, d_model))  # [1, 1, 128]
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling 
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True
        )

    def forward(self, x):
        # input x: [batch, seq_len, 12]
        batch_size = x.shape[0]
        x = self.action_embedding(x)  # [batch, seq_len, 128]
        # Apply the CLS token to the sequence head
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, 128]
        x = th.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, 128]
        x = self.pos_encoding(x)      # [batch, seq_len+1, 128]
        
        # Transformer Encoder
        x = self.transformer(x)  # [batch, seq_len+1, 128]
        
        # Attention Pooling using the CLS token
        cls_query = x[:, 0:1, :]  # [batch, 1, 128]
        pooled, _ = self.attn_pool(
            query=cls_query,
            key=x,
            value=x
        )  # pooled: [batch, 1, 128]
        
        return pooled.squeeze(1)  # [batch, 128]

class ActionLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,      
        hidden_dim: int = 128,   
        num_layers: int = 2,     
        dropout: float = 0.1       
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,      
            dropout=dropout if num_layers > 1 else 0 
        )
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        :param x: 输入动作序列 [batch_size, seq_len=100, input_dim=12]
        :return: 动作序列特征 [batch_size, hidden_dim]
        """
        output, (h_n, c_n) = self.lstm(x)
        return h_n[-1]  # [batch_size, hidden_dim]

class CustomCombinedExtractor_LSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 128,         
        lstm_hidden_dim: int = 64,         
        num_layers: int = 2,        
        lstm_input_dim: int = 12,            
        normalized_image: bool = False,    
        **kwargs
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = CustomResNet18(
                    subspace,
                    features_dim=cnn_output_dim,
                    normalized_image=normalized_image
                )
                total_concat_size += cnn_output_dim

            elif key == "action_sequence":
                extractors[key] = ActionLSTM(
                    input_dim = lstm_input_dim,
                    hidden_dim=lstm_hidden_dim,
                    num_layers=num_layers,
                    dropout = kwargs.get("dropout", 0.1)
                )
                total_concat_size += lstm_hidden_dim

            else:
                extractors[key] = nn.Flatten()
                flat_size = get_flattened_obs_dim(subspace)
                total_concat_size += flat_size

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, obs: TensorDict) -> th.Tensor:
        encoded_tensors = []
        for key, extractor in self.extractors.items():
            tensor = extractor(obs[key])
            encoded_tensors.append(tensor)
        return th.cat(encoded_tensors, dim=1)
