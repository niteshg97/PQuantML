"""Tests for convert_to_onnx / convert_to_onnx_fx.

Each test builds a small model (one PQ layer + ReLU where applicable), runs a
forward pass to initialise any running statistics, calls apply_final_compression
on every PQ module, exports to ONNX with convert_to_onnx(), and then verifies
that onnxruntime produces the same output as the PyTorch model.

The same check is repeated with bias=True and bias=False via parametrize.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

os.environ["KERAS_BACKEND"] = "torch"

import pquant  # noqa: E402
from pquant.core.torch.convert_to_onnx import (  # noqa: E402
    convert_to_onnx,
    convert_to_onnx_fx,
)
from pquant.layers import (  # noqa: E402
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQMultiheadAttention,
)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

ATOL = 1e-4  # float32 Gemm/Conv can differ by ~1 ULP; keep some slack


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = False
    return c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _apply_compression(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "apply_final_compression"):
            m.apply_final_compression()


def _onnx_run(model: nn.Module, x: torch.Tensor, input_shape: tuple, tmp_path) -> np.ndarray:
    """Export model → ONNX file in tmp_path, run with onnxruntime, return output."""
    path = str(tmp_path / "model.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.cpu().numpy()})[0]


def _onnx_run_fx(model: nn.Module, x: torch.Tensor, input_shape: tuple, tmp_path) -> np.ndarray:
    """FX-based export → ONNX, run with onnxruntime."""
    path = str(tmp_path / "model_fx.onnx")
    convert_to_onnx_fx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.cpu().numpy()})[0]


def _torch_out(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


# ---------------------------------------------------------------------------
# PQDense
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_dense_onnx(cfg, bias, tmp_path):
    IN, OUT = 16, 8
    model = nn.Sequential(
        PQDense(cfg, in_features=IN, out_features=OUT, bias=bias),
        nn.ReLU(),
    )
    x = torch.randn(4, IN)
    with torch.no_grad():
        model(x)  # warm-up (needed for any running stats)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(IN,), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQDense bias={bias}: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQConv2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_onnx(cfg, bias, tmp_path):
    IN_C, OUT_C, H, W = 3, 8, 8, 8
    model = nn.Sequential(
        PQConv2d(cfg, in_channels=IN_C, out_channels=OUT_C, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
    )
    x = torch.randn(2, IN_C, H, W)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(IN_C, H, W), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQConv2d bias={bias}: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQConv1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_conv1d_onnx(cfg, bias, tmp_path):
    IN_C, OUT_C, L = 4, 8, 16
    model = nn.Sequential(
        PQConv1d(cfg, in_channels=IN_C, out_channels=OUT_C, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
    )
    x = torch.randn(2, IN_C, L)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(IN_C, L), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQConv1d bias={bias}: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQBatchNorm2d
# ---------------------------------------------------------------------------


def test_batchnorm2d_onnx(cfg, tmp_path):
    C, H, W = 8, 4, 4
    model = nn.Sequential(
        PQBatchNorm2d(cfg, num_features=C),
        nn.ReLU(),
    )
    x = torch.randn(4, C, H, W)
    with torch.no_grad():
        model(x)
    _apply_compression(model)
    model.eval()  # switch BN to use running stats

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, H, W), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQBatchNorm2d: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQBatchNorm1d
# ---------------------------------------------------------------------------


def test_batchnorm1d_onnx(cfg, tmp_path):
    C, L = 8, 16
    model = nn.Sequential(
        PQBatchNorm1d(cfg, num_features=C),
        nn.ReLU(),
    )
    x = torch.randn(4, C, L)
    with torch.no_grad():
        model(x)
    _apply_compression(model)
    model.eval()

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, L), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQBatchNorm1d: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQAvgPool2d
# ---------------------------------------------------------------------------


def test_avgpool2d_onnx(cfg, tmp_path):
    C, H, W = 8, 8, 8
    model = nn.Sequential(
        PQAvgPool2d(cfg, kernel_size=2, stride=2),
    )
    x = torch.randn(2, C, H, W)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, H, W), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQAvgPool2d: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQAvgPool1d
# ---------------------------------------------------------------------------


def test_avgpool1d_onnx(cfg, tmp_path):
    C, L = 8, 16
    model = nn.Sequential(
        PQAvgPool1d(cfg, kernel_size=2, stride=2),
    )
    x = torch.randn(2, C, L)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, L), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQAvgPool1d: torch vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQMultiheadAttention  (uses FX converter; self-attention, batch_first=True)
# ---------------------------------------------------------------------------


class _SelfAttnModel(nn.Module):
    """Thin wrapper so FX tracing sees a single-input model."""

    def __init__(self, mha: PQMultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


@pytest.mark.parametrize("bias", [True, False])
def test_mha_onnx(cfg, bias, tmp_path):
    E, H, T = 16, 4, 8
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H, bias=bias, batch_first=True)
    model = _SelfAttnModel(mha)

    x = torch.randn(2, T, E)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run_fx(model, x, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_out, onnx_out, atol=ATOL, err_msg=f"PQMultiheadAttention bias={bias}: torch vs ONNX mismatch"
    )
