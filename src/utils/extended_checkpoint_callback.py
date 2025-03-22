import os
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter
import torch as th

class GraphCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        log_dir: str = None,  # Add a parameter for TensorBoard log directory
    ):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.log_dir = log_dir or save_path  # Use save_path as log_dir if not specified
        self.writer = None  # TensorBoard writer will be initialized in _on_training_start

    def _on_training_start(self) -> None:
        # Initialize the TensorBoard writer
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            # Assuming `example_input` is a tensor representing your input
            # You need to implement this method to return a tensor or a batch of tensors
            # that can be used as an input to your model.
            example_input = self._get_example_input()
            example_input = example_input.to(self.model.device)
            if example_input is not None:
                print("adding graph to tensorboard")
                self.model.policy.eval()
                self.writer.add_graph(self.model.policy, example_input)
                self.model.policy.train()
                print('adding operation ended')

    def _on_step(self) -> bool:
        super()._on_step()  # Call the parent method to handle model saving
        # Record parameters and gradients
        if self.writer is not None:
            for name, param in self.model.policy.named_parameters():
                self.writer.add_histogram(f"params/{name}", param, self.num_timesteps)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Close the writer when training ends
        if self.writer is not None:
            self.writer.close()

    def _get_example_input(self):
        # You need to implement this method based on your model's input format
        # This is necessary for `add_graph` to work.
        # Return a tensor or a batch of tensors that can be used as an input to your model.
        example_input = th.randn((1,) + self.model.policy.observation_space.shape)
        example_input.to(self.model.device)
        return example_input
