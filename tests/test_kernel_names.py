"""Host-only regressions for naming and profiling multiple TileLang backends."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import profiler, registry
from core.contract import KernelNameError
from core.drivers import tilelang


class KernelNameTests(unittest.TestCase):
    def setUp(self):
        self.compiled = Mock(prim_func=SimpleNamespace(attrs={"global_symbol": "tl_stream_v3"}))
        self.spec = SimpleNamespace(
            META=SimpleNamespace(name="stale_copy_name"),
            kernel=Mock(return_value=self.compiled),
            assemble=Mock(return_value=("input",)),
        )
        self.registration = patch.object(registry, "_BACKENDS", {"tl_stream_v3": self.spec})
        self.registration.start()
        self.addCleanup(self.registration.stop)

    def test_compile_and_run_use_registered_name(self):
        tilelang.compile(self.spec, "case")
        call = tilelang.make_call(self.spec, "case", "canonical")
        self.assertEqual(self.spec.kernel.call_count, 2)
        for invocation in self.spec.kernel.call_args_list:
            self.assertEqual(invocation.args, ("case",))
            self.assertEqual(invocation.kwargs, {"kernel_name": "tl_stream_v3"})
        self.spec.assemble.assert_called_once_with("case", "canonical")
        self.compiled.assert_not_called()
        self.assertIs(call(), self.compiled.return_value)
        self.compiled.assert_called_once_with("input")

    def test_wrong_or_missing_symbol_rejected_before_assemble(self):
        for attrs in ({"global_symbol": "main"}, {}, None):
            self.compiled.prim_func.attrs = attrs
            for run in (lambda: tilelang.compile(self.spec, "case"),
                        lambda: tilelang.make_call(self.spec, "case", "canonical")):
                with self.subTest(attrs=attrs), self.assertRaisesRegex(
                    KernelNameError, "expected 'tl_stream_v3'"
                ):
                    run()
        self.spec.assemble.assert_not_called()
        self.compiled.assert_not_called()

    def test_cached_result_is_checked_again(self):
        tilelang.compile(self.spec, "case")
        self.compiled.prim_func.attrs["global_symbol"] = "wrong_cached_symbol"
        with self.assertRaises(KernelNameError):
            tilelang.make_call(self.spec, "case", "canonical")
        self.spec.assemble.assert_not_called()

    def test_legacy_signature_is_a_contract_error(self):
        self.spec.kernel = lambda case: self.compiled
        with self.assertRaisesRegex(KernelNameError, "must accept"):
            tilelang.make_call(self.spec, "case", "canonical")
        self.spec.assemble.assert_not_called()

    def test_result_without_prim_func_is_rejected(self):
        self.spec.kernel.return_value = lambda: None
        with self.assertRaises(KernelNameError):
            tilelang.compile(self.spec, "case")

    def test_invalid_directory_symbol_rejected_before_build(self):
        with patch.object(registry, "_BACKENDS", {"bad-name": self.spec}):
            with self.assertRaises(KernelNameError):
                tilelang.compile(self.spec, "case")
        self.spec.kernel.assert_not_called()

    def test_registry_rejects_unregistered_or_ambiguous_module(self):
        for registered in ({}, {"one": self.spec, "two": self.spec}):
            with patch.object(registry, "_BACKENDS", registered):
                with self.assertRaises(ValueError):
                    registry.name_of(self.spec)


class ProfileNameTests(unittest.TestCase):
    def test_exact_symbols_disambiguate_prefixes_without_legacy_fallback(self):
        names = ["tl_stream", "tl_stream_v3", "npu_fa"]
        symbols = {n: f"{n}_kernel" for n in names[:2]}
        op_types = {n: "main_kernel" for n in names[:2]}
        op_types["npu_fa"] = "FlashAttentionScore"
        expected = {
            "tl_stream_kernel": "tl_stream",
            "tl_stream_v3_kernel": "tl_stream_v3",
            "main_kernel": None,
            "tl_stream_v30_kernel": None,
            "prefix_tl_stream_v3_kernel": None,
            "aclnnFlashAttentionScore": "npu_fa",
        }
        for symbol, backend in expected.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(
                    profiler._match_backend(symbol, names, op_types, symbols), backend
                )


if __name__ == "__main__":
    unittest.main()
