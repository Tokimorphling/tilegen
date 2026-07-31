"""CPU tests for the optional ComfyUI/comfy_kitchen integration."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

try:
    import comfy_kitchen
except ImportError:
    comfy_kitchen = None

from tilegen.comfyui import NODE_CLASS_MAPPINGS, backend


class ComfyUIIntegrationTests(unittest.TestCase):
    def test_drop_in_loader_exports_native_package_nodes(self) -> None:
        loader = Path(__file__).parents[1] / "comfyui" / "__init__.py"
        spec = importlib.util.spec_from_file_location("tilegen_test_loader", loader)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIs(module.NODE_CLASS_MAPPINGS, NODE_CLASS_MAPPINGS)
        self.assertIn("TileGenConvRotModelConfig", module.NODE_CLASS_MAPPINGS)

    def test_runtime_node_updates_environment_and_passes_model_through(self) -> None:
        node = NODE_CLASS_MAPPINGS["TileGenConvRotModelConfig"]()
        model = object()
        with patch.dict(os.environ, {}, clear=True):
            result = node.configure(model, "fht", 5120, "native", 768, False)
            self.assertEqual(result, (model,))
            self.assertEqual(os.environ["TILEGEN_CONVROT_BACKEND"], "fht")
            self.assertEqual(os.environ["TILEGEN_FHT_MIN_K"], "5120")
            self.assertEqual(os.environ["TILEGEN_FHT_IMPL"], "native")
            self.assertEqual(os.environ["TILEGEN_INT8_TEMP_MB"], "768")
            self.assertEqual(os.environ["TILEGEN_DIAGNOSTICS"], "0")

    @unittest.skipIf(comfy_kitchen is None, "comfy-kitchen optional dependency is not installed")
    def test_install_registers_selective_backend_first(self) -> None:
        from comfy_kitchen.registry import registry

        with patch.dict(os.environ, {"TILEGEN_CONVROT_BACKEND": "auto"}):
            self.assertTrue(backend.install())
            self.assertEqual(registry._priority[0], "tilegen")
            constraints = registry.get_constraints("tilegen", "int8_linear")
            self.assertIsNotNone(constraints)
            self.assertIn(backend._fht_applicable, constraints.call_rules)

            result = backend._fht_applicable({
                "x": torch.zeros(2, 256, dtype=torch.float16),
                "weight": torch.zeros(4, 256, dtype=torch.int8),
                "convrot": True,
                "convrot_groupsize": 256,
            })
            self.assertFalse(result.success)
            self.assertEqual(result.failed_param, "x")

    @unittest.skipIf(comfy_kitchen is None, "comfy-kitchen optional dependency is not installed")
    def test_eager_compatibility_accepts_input_act(self) -> None:
        from comfy_kitchen.backends.eager.quantization import (
            int8_linear as eager_int8_linear,
        )

        torch.manual_seed(7)
        x = torch.randn(3, 256, dtype=torch.float32)
        weight = torch.randint(-20, 20, (8, 256), dtype=torch.int8)
        weight_scale = torch.full((8,), 0.01, dtype=torch.float32)
        bias = torch.randn(8, dtype=torch.float32)
        kwargs = {
            "bias": bias,
            "out_dtype": torch.float32,
            "convrot": True,
            "convrot_groupsize": 256,
            "input_act": "gelu_tanh",
        }
        with patch.dict(os.environ, {"TILEGEN_CONVROT_BACKEND": "eager"}):
            actual = backend.int8_linear(x, weight, weight_scale, **kwargs)
        expected = eager_int8_linear(x, weight, weight_scale, **kwargs)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
