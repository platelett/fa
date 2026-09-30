"""Host-only regressions for generic callables and standard operator projects."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core import registry
from core.drivers import ascendc, external


class ExternalTests(unittest.TestCase):
    def test_registration_and_precompile_selection(self):
        prebuilt = SimpleNamespace(META=SimpleNamespace(type="external"))
        source = SimpleNamespace(META=prebuilt.META, build=Mock())
        self.assertIs(registry.driver_for(prebuilt.META), external)
        self.assertFalse(registry.needs_compile(prebuilt))
        self.assertTrue(registry.needs_compile(source))
        self.assertTrue(
            registry.needs_compile(SimpleNamespace(META=SimpleNamespace(type="ascendc")))
        )

    def test_removed_torch_type_is_rejected(self):
        with self.assertRaises(KeyError):
            registry.driver_for(SimpleNamespace(type="torch"))

    def test_npu_reference_uses_external_without_changing_call_arguments(self):
        from backends.npu_fa import backend

        self.assertEqual(backend.META.type, "external")
        self.assertIs(registry.driver_for(backend.META), external)
        self.assertFalse(registry.needs_compile(backend))
        self.assertTrue(backend.META.is_reference)
        canonical = {name: object() for name in ("q", "k", "v")}
        case = SimpleNamespace(heads_q=12, scale=0.125)
        output = object()
        fusion = Mock(return_value=(output,))
        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu_fusion_attention=fusion)}):
            external.compile(backend, case)
            call = external.make_call(backend, case, canonical)
            fusion.assert_not_called()
            self.assertIs(call(), output)
            self.assertIs(call(), output)
        self.assertEqual(fusion.call_count, 2)
        fusion.assert_called_with(
            canonical["q"],
            canonical["k"],
            canonical["v"],
            case.heads_q,
            padding_mask=None,
            atten_mask=None,
            scale=case.scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
        )

    def test_optional_build_and_opaque_argument_object(self):
        args = {"q": "tensor", "custom": 123}
        spec = SimpleNamespace(adapt=Mock(return_value=args), call=Mock(return_value="out"))
        external.compile(spec, "case")
        call = external.make_call(spec, "case", "canonical")
        spec.adapt.assert_called_once_with("case", "canonical")
        self.assertEqual(call(), "out")
        self.assertEqual(call(), "out")
        self.assertEqual(spec.call.call_count, 2)
        spec.call.assert_called_with(args)
        spec.build = Mock()
        external.compile(spec, "case")
        external.make_call(spec, "case", "canonical")
        self.assertEqual(spec.build.call_count, 2)

    def test_generic_prepare_defers_adaptation_until_inputs_exist(self):
        spec = SimpleNamespace(
            META=SimpleNamespace(type="external"),
            adapt=Mock(return_value="args"),
            call=Mock(return_value="out"),
        )
        bind = registry.prepare_call(spec, "case")
        spec.adapt.assert_not_called()
        call = bind("canonical")
        spec.adapt.assert_called_once_with("case", "canonical")
        self.assertEqual(call(), "out")

    def test_bench_prepares_packages_before_npu_inputs(self):
        from core import bench
        from problem import canonical

        events = []
        spec = SimpleNamespace(
            META=SimpleNamespace(type="ascendc", op_type="Test", is_reference=True)
        )

        def prepare(spec, case):
            events.append("load")

            def bind(inputs):
                events.append("adapt")
                return lambda: None

            return bind

        def inputs(case, seed):
            events.append("inputs")
            return {}

        with (
            patch.object(registry, "get", return_value=spec),
            patch.object(registry, "prepare_call", side_effect=prepare),
            patch.object(canonical, "build_canonical", side_effect=inputs),
            patch.object(bench.torch, "npu", SimpleNamespace(synchronize=Mock()), create=True),
            patch.object(bench.profiler, "profile_case_us", return_value={"test": 1.0}),
            patch.dict("os.environ"),
        ):
            bench.process_case(
                0,
                0,
                session_dir="unused",
                backend_names=["test"],
                ref_name="test",
                iters=1,
                do_check=False,
                seed=0,
                atol=0,
                rtol=0,
            )
        self.assertEqual(events, ["load", "inputs", "adapt"])

    def test_bench_selects_source_external_and_standard_projects(self):
        from core import bench

        specs = {
            name: SimpleNamespace(META=SimpleNamespace(type=kind))
            for name, kind in (
                ("prebuilt", "external"),
                ("source", "external"),
                ("project", "ascendc"),
                ("torch_op", "external"),
            )
        }
        specs["source"].build = Mock()
        seen = []

        def collect(tasks, worker):
            seen.extend(tasks)
            return []

        with (
            patch.object(registry, "_BACKENDS", specs),
            patch.object(bench.schedule, "parallel_compile", side_effect=collect),
        ):
            bench.compile_phase(list(specs), [0, 1])
        self.assertEqual(seen, [("source", 0), ("source", 1), ("project", 0), ("project", 1)])


class AscendCProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "CMakeLists.txt").write_text("# synthetic host-test project\n")
        for name in ("op_host", "op_kernel"):
            (self.root / name).mkdir()
            (self.root / name / "op.cpp").write_text("// structure fixture, not a CANN operator\n")

    def spec(self, command=()):
        project = ascendc.Project(self.root, ("build/libop.so",), command)
        operator = Mock(return_value="canonical output")
        return SimpleNamespace(
            project=Mock(return_value=project),
            load=Mock(return_value=operator),
            adapt=Mock(return_value=("q", "k", "v")),
        )

    def test_legacy_generic_backend_is_rejected(self):
        spec = SimpleNamespace(adapt=Mock(), call=Mock())
        with self.assertRaisesRegex(TypeError, 'type="external"'):
            ascendc.compile(spec, "case")
        spec.adapt.assert_not_called()
        spec.call.assert_not_called()

    def test_prepare_loads_without_adapting_or_launching(self):
        (self.root / "build").mkdir()
        (self.root / "build/libop.so").write_bytes(b"test fixture")
        spec = self.spec()
        spec.META = SimpleNamespace(type="ascendc")
        bind = registry.prepare_call(spec, "case")
        spec.load.assert_called_once()
        spec.adapt.assert_not_called()
        spec.load.return_value.assert_not_called()
        call = bind("canonical")
        spec.adapt.assert_called_once_with("case", "canonical")
        self.assertEqual(call(), "canonical output")

    def test_structural_validation_precedes_build(self):
        (self.root / "op_host/op.cpp").unlink()
        with patch.object(ascendc.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "op_host"):
                ascendc.compile(self.spec(("unused",)), "case")
            run.assert_not_called()

    def test_explicit_build_outputs_load_and_call(self):
        command = (
            sys.executable,
            "-c",
            "from pathlib import Path; Path('build').mkdir(exist_ok=True); "
            "Path('build/libop.so').write_bytes(b'host-test artifact, not ELF')",
        )
        spec = self.spec(command)
        ascendc.compile(spec, "case")
        self.assertTrue((self.root / "build/libop.so").is_file())
        spec.load.assert_not_called()  # CPU precompile never loads the operator.
        call = ascendc.make_call(spec, "case", "canonical")
        spec.load.assert_called_once_with("case", spec.project.return_value)
        spec.adapt.assert_called_once_with("case", "canonical")
        self.assertEqual(call(), "canonical output")
        spec.load.return_value.assert_called_once_with("q", "k", "v")

    def test_prebuilt_project_requires_declared_outputs(self):
        spec = self.spec()
        with self.assertRaises(FileNotFoundError):
            ascendc.make_call(spec, "case", "canonical")
        spec.load.assert_not_called()
        (self.root / "build").mkdir()
        (self.root / "build/libop.so").write_bytes(b"fixture")
        with patch.object(ascendc.subprocess, "run") as run:
            ascendc.compile(spec, "case")
            run.assert_not_called()

    def test_failed_build_never_loads_or_runs(self):
        spec = self.spec((sys.executable, "-c", "raise SystemExit(3)"))
        with self.assertRaises(subprocess.CalledProcessError):
            ascendc.make_call(spec, "case", "canonical")
        spec.load.assert_not_called()
        spec.adapt.assert_not_called()

    def test_invalid_recipe_and_escaping_artifacts_rejected(self):
        for project in (
            ascendc.Project(self.root, ("../escape.so",)),
            ascendc.Project(self.root, ("/absolute.so",)),
            ascendc.Project(self.root, ()),
            ascendc.Project(self.root, ("build/lib.so",), "bash build.sh"),
        ):
            with self.subTest(project=project), self.assertRaises((ValueError, TypeError)):
                project.validate()

    def test_binding_and_argument_contracts_checked(self):
        (self.root / "build").mkdir()
        (self.root / "build/libop.so").write_bytes(b"fixture")
        spec = self.spec()
        spec.load.return_value = None
        with self.assertRaisesRegex(TypeError, "callable binding"):
            ascendc.make_call(spec, "case", "canonical")
        spec.load.return_value = Mock()
        spec.adapt.return_value = {"q": "not a tuple"}
        with self.assertRaisesRegex(TypeError, "positional-argument tuple"):
            ascendc.make_call(spec, "case", "canonical")


if __name__ == "__main__":
    unittest.main()
