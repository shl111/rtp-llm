# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test a torch_npu-native implementation of sigmoid_gate_scale_add against
GPU-collected golden data.

- Backend: pure torch ops running on the NPU device (no rtp-llm Triton kernel).
  This is an independent reference/fallback implementation, mirroring the
  kernel's fp32-intermediate semantics:
      experts[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + experts[t, :]
- Compared against the same golden data used by the Triton-kernel test
  (``test_npu_sigmoid_gate_scale_add_triton_gpu_golden.py``).

Golden layout (in each ``sample_moe/sigmoid_gate_scale_add_triton/*.pt``)
------------------------------------------------------------------------
  inputs:
    gate     : (T, 1)   float16 — scalar gate per token
    shared   : (T, H)   float16 — shared expert MLP output
    experts  : (T, H)   float16 — routed experts output (initial value)
  outputs   : (T, H)    float16 — golden kernel result (= final experts)
  inplace_outputs:
    experts  : (T, H)   float16 — same values as outputs (separate buffer)

Golden sanity: before running on NPU, each case is cross-checked with a CPU
reference (``sigmoid(gate) * shared + experts`` in fp32).
"""

import os
import unittest

import torch

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SAMPLE_ROOT = os.path.join(_WORKSPACE_ROOT, "sample_moe", "sigmoid_gate_scale_add_triton")


def sigmoid_gate_scale_add_torch(gate, shared, experts):
    """torch_npu-native implementation (fp32 intermediate, in-place on experts).

    Matches the GPU Triton kernel semantics: compute in fp32, round once on
    write-back. (rtp-llm's own Ascend fallback uses ``experts.add_(sigmoid(gate)
    * shared)`` which is fp16-intermediate; this variant keeps fp32 to match the
    fp32-intermediate golden.)
    """
    experts.copy_(
        (torch.sigmoid(gate.float()) * shared.float() + experts.float()).to(shared.dtype)
    )
    return experts


def _load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"golden file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _cpu_reference(gate, shared, experts):
    result = torch.sigmoid(gate.float()) * shared.float() + experts.float()
    return result.to(shared.dtype)


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestSigmoidGateScaleAddTorchGpuGolden(unittest.TestCase):
    """Compare the torch_npu implementation against GPU golden data."""

    rtol = 1e-2
    atol = 1e-2

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), "output shape mismatch")
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, filename):
        path = os.path.join(_SAMPLE_ROOT, filename)
        data = _load_pt(path)
        inputs = data["inputs"]
        outputs = data.get("inplace_outputs", {})

        gate = inputs["gate"]
        shared = inputs["shared"]
        experts = inputs["experts"]  # initial value
        experts_expected = outputs["experts"]

        T, H = shared.shape
        self.assertEqual(tuple(gate.shape), (T, 1))
        self.assertEqual(tuple(experts.shape), (T, H))
        self.assertEqual(tuple(experts_expected.shape), (T, H))

        ref = _cpu_reference(gate, shared, experts)
        self.assertTrue(
            torch.allclose(ref.float(), experts_expected.float(), rtol=1e-2, atol=1e-2),
            msg="CPU ref mismatch: "
            f"max_abs_diff={(ref.float() - experts_expected.float()).abs().max().item():.6f}",
        )

        experts_actual = experts.clone().npu()
        ret = sigmoid_gate_scale_add_torch(gate.npu(), shared.npu(), experts_actual)
        torch.npu.synchronize()

        self.assertTrue(ret is experts_actual, "must modify experts in-place")
        self.assertTensorClose(experts_actual, experts_expected)

    def test_T1_H2048(self):
        self._run_case("T1_H2048.pt")

    def test_T16_H2048(self):
        self._run_case("T16_H2048.pt")

    def test_T2047_H2048(self):
        self._run_case("T2047_H2048.pt")


if __name__ == "__main__":
    unittest.main()
