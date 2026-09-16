# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ROOT / "turbo_physai" / "operators"


class OperatorCompatibilityTest(unittest.TestCase):
    def test_kernel_adapters_exist_in_operator_layer(self):
        expected = {
            "bev_pool.py": {
                "bev_pool",
                "bev_pool_prepare",
                "bev_pool_prepare_geometry",
            },
            "voxelization.py": {
                "voxelize",
                "voxelization_forward",
                "dynamic_voxelize",
                "hard_voxelize",
            },
            "sparse_conv.py": {
                "get_indice_pairs",
                "indice_conv_fp32",
                "indice_conv_backward_fp32",
                "indice_conv_half",
                "indice_conv_backward_half",
                "fused_indice_conv_fp32",
                "fused_indice_conv_half",
                "indice_maxpool_fp32",
                "indice_maxpool_backward_fp32",
                "indice_maxpool_half",
                "indice_maxpool_backward_half",
            },
        }
        for filename, names in expected.items():
            tree = ast.parse((OPERATORS / filename).read_text(encoding="utf-8"))
            definitions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertTrue(names <= definitions, (filename, names - definitions))

    def test_every_native_binding_has_an_operator_wrapper(self):
        binding_sources = [
            ROOT / "turbo_physai" / "csrc" / "pybind.cpp",
            ROOT / "kernel" / "voxelization" / "src" / "voxelization.cpp",
            ROOT / "kernel" / "sparse_conv" / "src" / "all.cc",
            ROOT / "kernel" / "bev_pool" / "src" / "bev_pool_cpu.cpp",
        ]
        bindings = set()
        for path in binding_sources:
            source = path.read_text(encoding="utf-8")
            bindings.update(re.findall(r'm\.def\("([A-Za-z0-9_]+)"', source))
        trees = [
            ast.parse(path.read_text(encoding="utf-8"))
            for path in OPERATORS.glob("*.py")
        ]
        wrappers = {
            node.name
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(bindings <= wrappers, sorted(bindings - wrappers))

    def test_operator_modules_use_bundled_extension_lazily(self):
        for filename in ("bev_pool.py", "voxelization.py", "sparse_conv.py"):
            source = (OPERATORS / filename).read_text(encoding="utf-8")
            self.assertIn("from turbo_physai import ops", source)


if __name__ == "__main__":
    unittest.main()
