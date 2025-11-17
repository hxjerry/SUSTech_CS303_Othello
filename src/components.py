from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Dict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = 64
BN_EPS = 1e-5
_UNPACKER_MASK = (np.uint64(1) << np.arange(64, dtype=np.uint64)).reshape(1, 1, 8, 8)


def bitboards_to_tensor(player: np.ndarray | int, opponent: np.ndarray | int) -> np.ndarray:
	"""Convert bitboard integers into float32 tensors shaped (batch, 2, 8, 8)."""
	player_bits = np.atleast_1d(np.asarray(player, dtype=np.uint64))
	opponent_bits = np.atleast_1d(np.asarray(opponent, dtype=np.uint64))
	if player_bits.shape != opponent_bits.shape:
		raise ValueError("Player and opponent bitboards must share the same shape")

	stacked = np.stack((player_bits, opponent_bits), axis=1)  # (batch, 2)
	batch = stacked.shape[0]
	board = ((stacked.reshape(batch, 2, 1, 1) & _UNPACKER_MASK) != 0).astype(np.float32)
	return board


def numpy_state_dict_from_torch(state_dict: Mapping[str, Any]) -> Dict[str, np.ndarray]:
	"""Helper to detach a PyTorch state dict into numpy arrays."""
	return {key: value.detach().cpu().numpy() for key, value in state_dict.items()}


@dataclass(frozen=True)
class ConvBlockParams:
	weight: np.ndarray
	bias: np.ndarray
	bn_weight: np.ndarray
	bn_bias: np.ndarray
	bn_running_mean: np.ndarray
	bn_running_var: np.ndarray
	padding: int


@dataclass(frozen=True)
class FCBlockParams:
	weight: np.ndarray
	bias: np.ndarray
	bn_weight: np.ndarray
	bn_bias: np.ndarray
	bn_running_mean: np.ndarray
	bn_running_var: np.ndarray


@dataclass(frozen=True)
class LinearParams:
	weight: np.ndarray
	bias: np.ndarray


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, padding: int) -> np.ndarray:
	if padding:
		x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

	kernel_h, kernel_w = weight.shape[-2:]
	patches = sliding_window_view(x, (kernel_h, kernel_w), axis=(-2, -1))
	out = np.tensordot(patches, weight, axes=([1, 4, 5], [1, 2, 3]))
	out = np.moveaxis(out, -1, 1)
	if bias is not None:
		out += bias[None, :, None, None]
	return out


def _batch_norm_2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
				   running_mean: np.ndarray, running_var: np.ndarray) -> np.ndarray:
	inv_std = 1.0 / np.sqrt(running_var + BN_EPS)
	return ((x - running_mean[None, :, None, None]) * inv_std[None, :, None, None]) * weight[None, :, None, None] + bias[None, :, None, None]


def _batch_norm_1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
				   running_mean: np.ndarray, running_var: np.ndarray) -> np.ndarray:
	inv_std = 1.0 / np.sqrt(running_var + BN_EPS)
	return ((x - running_mean[None, :]) * inv_std[None, :]) * weight[None, :] + bias[None, :]


def _relu(x: np.ndarray) -> np.ndarray:
	return np.maximum(x, 0.0)


def _conv_block_forward(x: np.ndarray, params: ConvBlockParams) -> np.ndarray:
	y = _conv2d(x, params.weight, params.bias, params.padding)
	y = _batch_norm_2d(y, params.bn_weight, params.bn_bias, params.bn_running_mean, params.bn_running_var)
	return _relu(y)


def _fc_block_forward(x: np.ndarray, params: FCBlockParams) -> np.ndarray:
	y = x @ params.weight.T + params.bias
	y = _batch_norm_1d(y, params.bn_weight, params.bn_bias, params.bn_running_mean, params.bn_running_var)
	return _relu(y)


def _linear_forward(x: np.ndarray, params: LinearParams) -> np.ndarray:
	return x @ params.weight.T + params.bias


class NumpyOthelloNet:
	def __init__(self, state_dict: Mapping[str, np.ndarray]):
		self.conv1 = self._build_conv_block("conv1", state_dict, padding=1)
		self.conv2 = self._build_conv_block("conv2", state_dict, padding=1)
		self.conv3 = self._build_conv_block("conv3", state_dict, padding=0)
		self.conv4 = self._build_conv_block("conv4", state_dict, padding=0)

		self.fc1 = self._build_fc_block("fc1", state_dict)
		self.fc_policy = self._build_linear("fc_policy", state_dict)
		self.fc_value = self._build_linear("fc_value", state_dict)

	@staticmethod
	def _build_conv_block(prefix: str, state_dict: Mapping[str, np.ndarray], padding: int) -> ConvBlockParams:
		weight = state_dict[f"{prefix}.conv.weight"].astype(np.float32)
		bias = state_dict[f"{prefix}.conv.bias"].astype(np.float32)
		bn_weight = state_dict[f"{prefix}.bn.weight"].astype(np.float32)
		bn_bias = state_dict[f"{prefix}.bn.bias"].astype(np.float32)
		running_mean = state_dict[f"{prefix}.bn.running_mean"].astype(np.float32)
		running_var = state_dict[f"{prefix}.bn.running_var"].astype(np.float32)
		return ConvBlockParams(weight, bias, bn_weight, bn_bias, running_mean, running_var, padding)

	@staticmethod
	def _build_fc_block(prefix: str, state_dict: Mapping[str, np.ndarray]) -> FCBlockParams:
		weight = state_dict[f"{prefix}.fc.weight"].astype(np.float32)
		bias = state_dict[f"{prefix}.fc.bias"].astype(np.float32)
		bn_weight = state_dict[f"{prefix}.bn.weight"].astype(np.float32)
		bn_bias = state_dict[f"{prefix}.bn.bias"].astype(np.float32)
		running_mean = state_dict[f"{prefix}.bn.running_mean"].astype(np.float32)
		running_var = state_dict[f"{prefix}.bn.running_var"].astype(np.float32)
		return FCBlockParams(weight, bias, bn_weight, bn_bias, running_mean, running_var)

	@staticmethod
	def _build_linear(prefix: str, state_dict: Mapping[str, np.ndarray]) -> LinearParams:
		weight = state_dict[f"{prefix}.weight"].astype(np.float32)
		bias = state_dict[f"{prefix}.bias"].astype(np.float32)
		return LinearParams(weight, bias)

	def forward(self, board_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if board_tensor.ndim != 4 or board_tensor.shape[1:] != (2, 8, 8):
			raise ValueError("Input must have shape (batch, 2, 8, 8)")

		x = board_tensor.astype(np.float32)
		x = _conv_block_forward(x, self.conv1)
		x = _conv_block_forward(x, self.conv2)
		x = _conv_block_forward(x, self.conv3)
		x = _conv_block_forward(x, self.conv4)

		x = x.reshape(x.shape[0], -1)
		x = _fc_block_forward(x, self.fc1)

		policy = _linear_forward(x, self.fc_policy)
		value = np.tanh(_linear_forward(x, self.fc_value))
		return policy, value

	def predict_from_bitboards(self, player: np.ndarray | int, opponent: np.ndarray | int) -> Tuple[np.ndarray, np.ndarray]:
		board = bitboards_to_tensor(player, opponent)
		return self.forward(board)


def _torch_smoke_test() -> None:
	try:
		import torch
	except Exception as exc:  # pragma: no cover - optional test helper
		raise RuntimeError("Torch smoke test requires PyTorch to be installed.") from exc

	import sys
	from pathlib import Path

	repo_root = Path(__file__).resolve().parent.parent
	repo_str = str(repo_root)
	if repo_str not in sys.path:
		sys.path.insert(0, repo_str)

	from src.model import OthelloNet as TorchOthelloNet  # type: ignore

	torch_model = TorchOthelloNet().eval()
	torch_state = torch_model.state_dict()
	numpy_state = {k: v.detach().cpu().numpy() for k, v in torch_state.items()}
	numpy_model = NumpyOthelloNet(numpy_state)

	sample_player = np.uint64(0x0000000810000000)
	sample_opponent = np.uint64(0x0000001008000000)
	board = bitboards_to_tensor(sample_player, sample_opponent)

	with torch.inference_mode():
		torch_policy, torch_value = torch_model(torch.from_numpy(board))

	numpy_policy, numpy_value = numpy_model.forward(board)
	if not (np.allclose(numpy_policy, torch_policy.detach().cpu().numpy(), atol=1e-5) and
			np.allclose(numpy_value, torch_value.detach().cpu().numpy(), atol=1e-5)):
		raise AssertionError("NumPy and PyTorch outputs do not match")


if __name__ == "__main__":
	_torch_smoke_test()
	print("NumPy inference matches PyTorch model for the sample input.")
