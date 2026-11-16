"""First-call public GC control; original test assertions remain unchanged."""
import argparse
import ast
import contextlib
import functools
import gc
import hashlib
import inspect
import json
import pathlib
import re
import sys
import unittest
import types
import weakref

EXPECTED_AUTOGRAD_SHA = "088fd54272d5df9087e770433dfee556add8d56d8371dff3906960c910eeb71e"
EXPECTED_COMPILED_SHA = "ccc37bdaa8ae3ff36258fabca15f1427f55b775f86407ac832ae284ce4da812c"
EXPECTED_BUILDER_METHOD_AST_SHA = "744639f675cc9edee7aac4bc3d925cb775d55d7cb15b22be4ac8559f441aebce"
EXPECTED_BOXED_PATCH_SHA = "e5ce2370fea63451a4d6a5d45a8b377ade99d4c5520cbbfac597039a53550fff"
EXPECTED_INTERFACE_PATCH_SHA = "ab9f1773304575153006135ccf8f11d4a70322750a590f9c767d9eaea7844c95"
EXPECTED_APPROVED_COMPILED_SHA = "b2d8271e1577a9a974d3fa81214807266d3a51a27b51077458043ff44dbeb3b4"
TEST_NAME = "test_custom_function_cycle"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def method_ast(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VariableBuilder")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "wrap_listlike")


def ast_digest(node):
    return digest(ast.dump(node, include_attributes=False).encode())


def apply_single_file_diff(source, patch, relative_path="torch/_dynamo/variables/builder.py"):
    lines = patch.splitlines(keepends=True)
    target = "--- a/" + relative_path
    if sum(line.rstrip("\n") == target for line in lines) != 1:
        raise RuntimeError("The patch must identify the expected source file exactly once")
    hunks = []
    active = False
    for line in lines:
        if line.startswith("--- a/"):
            active = line.rstrip("\n") == target
        elif active and line.startswith("@@ "):
            hunks.append([[], []])
        elif active and hunks and line.startswith((" ", "-", "+")) and not line.startswith(("--- ", "+++ ")):
            if line[0] in " -":
                hunks[-1][0].append(line[1:])
            if line[0] in " +":
                hunks[-1][1].append(line[1:])
    if not hunks:
        raise RuntimeError("No source patch hunk found")
    for old, new in hunks:
        old_text = "".join(old)
        new_text = "".join(new)
        if source.count(old_text) != 1:
            raise RuntimeError("Source patch context must occur exactly once")
        source = source.replace(old_text, new_text, 1)
    return source



def capture_counter_snapshot(counters):
    """Read existing process counters without inserting or clearing entries."""
    keys = (
        ("compiled_autograd", "captures"),
        ("compiled_autograd", "compiles"),
        ("frames", "total"),
        ("frames", "ok"),
        ("stats", "calls_captured"),
        ("stats", "unique_graphs"),
        ("aot_autograd", "total"),
        ("aot_autograd", "ok"),
        ("aot_autograd", "not_ok"),
        ("inductor", "fxgraph_cache_hit"),
        ("inductor", "fxgraph_cache_miss"),
        ("inductor", "fxgraph_cache_bypass"),
    )
    result = {}
    for group, key in keys:
        value = counters.get(group, {}).get(key, 0)
        if type(value) is int:
            result[group + "." + key] = value
    return result


def observe_weak_tensor_refs(torch, reports):
    """Observe owners after a real post-collection expiration check fails."""
    original = torch._C._WeakTensorRef
    count = 0

    def ownership_report(tensor_ref, context_ref, impl, ordinal):
        tensor = tensor_ref()
        context = context_ref() if context_ref is not None else None
        ownership_counts = {
            "measurement_phase": "after_real_false_before_object_inventory",
            "tensor_impl_use_count": tensor._use_count() if tensor is not None else None,
            "tensor_python_refcount": sys.getrefcount(tensor) if tensor is not None else None,
            "context_python_refcount": sys.getrefcount(context) if context is not None else None,
            "python_refcounts_include_observer_references": True,
            "tensor_gc_tracked": gc.is_tracked(tensor) if tensor is not None else None,
            "context_gc_tracked": gc.is_tracked(context) if context is not None else None,
            "tensor_gc_finalized": gc.is_finalized(tensor) if tensor is not None else None,
            "context_gc_finalized": gc.is_finalized(context) if context is not None else None,
            "gc_frozen_object_count": gc.get_freeze_count(),
            "gc_debug_saveall": bool(gc.get_debug() & gc.DEBUG_SAVEALL),
            "inductor_compile_threads": torch._inductor.config.compile_threads,
            "inductor_worker_start_method": torch._inductor.config.worker_start_method,
        }
        snapshot = gc.get_objects()
        roots = []
        temporary = {id(snapshot), id(roots), id(reports)}
        if tensor is not None:
            roots.append(("output", tensor))
        if context is not None:
            roots.append(("autograd_context", context))
        for obj in snapshot:
            if type(obj) is torch.Tensor and obj is not tensor:
                if obj._cdata == impl or (context is not None and obj.grad_fn is context):
                    roots.append(("tensor_alias", obj))
        temporary.update(id(entry) for entry in roots)
        nodes, edges, indices = [], [], {}
        temporary.update((id(nodes), id(edges), id(indices)))

        def visit(obj, depth):
            key = id(obj)
            if key in indices:
                return indices[key]
            if len(nodes) >= 64:
                return None
            index = len(nodes)
            indices[key] = index
            row = {"index": index, "type": type(obj).__module__ + "." + type(obj).__qualname__}
            nodes.append(row)
            temporary.add(id(row))
            if type(obj) is torch.Tensor:
                row["same_tensor_impl"] = obj._cdata == impl
            elif isinstance(obj, dict):
                known_keys = {"foo", "meta", "example_value", "val", "tensor_meta",
                              "steal_arg", "stack_trace"}
                row["known_torch_metadata_keys"] = sorted(k for k in obj if type(k) is str and k in known_keys)
                if "__builtins__" in obj and "__name__" in obj:
                    row["module_namespace"] = True
                    return index
            elif isinstance(obj, torch.fx.Node):
                row["op"] = obj.op
            if depth >= 5:
                row["depth_limit"] = True
                return index
            owners = gc.get_referrers(obj)
            temporary.add(id(owners))
            kept = 0
            for owner in owners:
                if id(owner) in temporary or type(owner) is ObservedWeakTensorRef:
                    continue
                if isinstance(owner, types.FrameType) and owner.f_code.co_filename == __file__:
                    continue
                owner_index = visit(owner, depth + 1)
                if owner_index is not None:
                    edges.append({"owner": owner_index, "owned": index})
                    kept += 1
                if kept >= 8:
                    row["owner_limit"] = True
                    break
            return index

        root_rows = [{"role": role, "index": visit(obj, 0)} for role, obj in roots]
        return {"reference_ordinal": ordinal, "expired_result": False,
                "observation_phase": "after_real_result_before_original_assertion",
                "python_tensor_alive": tensor is not None,
                "python_context_alive": context is not None,
                "ownership_counts": ownership_counts,
                "capture_counters": capture_counter_snapshot(torch._dynamo.utils.counters),
                "roots": root_rows, "nodes": nodes, "edges": edges,
                "limits": {"nodes": 64, "owners_per_node": 8, "depth": 5},
                "extra_gc_calls": 0}

    class ObservedWeakTensorRef:
        def __init__(self, tensor):
            nonlocal count
            count += 1
            self.ordinal = count
            self.inner = original(tensor)
            self.tensor_ref = weakref.ref(tensor)
            context = tensor.grad_fn
            self.context_ref = weakref.ref(context) if context is not None else None
            self.impl = tensor._cdata
            self.checks = 0

        def expired(self):
            value = self.inner.expired()
            self.checks += 1
            if self.checks == 2 and not value:
                try:
                    reports.append(ownership_report(self.tensor_ref, self.context_ref,
                                                    self.impl, self.ordinal))
                except Exception as error:
                    reports.append({"reference_ordinal": self.ordinal,
                                    "expired_result": value,
                                    "observation_error": type(error).__name__})
            return value

    torch._C._WeakTensorRef = ObservedWeakTensorRef
    return original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("baseline", "boxed"), required=True)
    parser.add_argument("--mode", choices=("automatic", "direct-inductor"), required=True)
    parser.add_argument("--autograd-source", type=pathlib.Path, required=True)
    parser.add_argument("--compiled-autograd-source", type=pathlib.Path, required=True)
    parser.add_argument("--boxed-patch", type=pathlib.Path, required=True)
    parser.add_argument("--expected-boxed-patch-sha256", default=EXPECTED_BOXED_PATCH_SHA)
    parser.add_argument("--builder-reference", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--observe-reference-owners", action="store_true")
    parser.add_argument("--wrapper-cohort", choices=("original", "approved"), default="original")
    parser.add_argument("--test-interface-patch", type=pathlib.Path)
    args = parser.parse_args()

    import torch
    import torch._dynamo.variables.builder as builder_module
    import torch._inductor as inductor
    from torch._dynamo import compiled_autograd, config
    from torch._dynamo.backends.debugging import aot_eager
    from torch._dynamo.utils import counters
    from torch.autograd import Function
    from torch.testing._internal.common_utils import TestCase, disable_gc, IS_S390X
    from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_XPU_AND_TRITON

    autograd_source = args.autograd_source.read_text()
    compiler_source = args.compiled_autograd_source.read_text()
    patch = args.boxed_patch.read_text()
    if digest(args.autograd_source.read_bytes()) != EXPECTED_AUTOGRAD_SHA:
        raise RuntimeError("Original autograd test source hash differs")
    if digest(args.compiled_autograd_source.read_bytes()) != EXPECTED_COMPILED_SHA:
        raise RuntimeError("Original compiled-autograd test source hash differs")
    if digest(args.boxed_patch.read_bytes()) != args.expected_boxed_patch_sha256:
        raise RuntimeError("Owned boxed production patch hash differs")

    interface_patch_sha = None
    if args.wrapper_cohort == "approved":
        if args.test_interface_patch is None:
            raise RuntimeError("The approved wrapper requires its exact interface patch")
        interface_patch = args.test_interface_patch.read_text()
        interface_patch_sha = digest(args.test_interface_patch.read_bytes())
        if interface_patch_sha != EXPECTED_INTERFACE_PATCH_SHA:
            raise RuntimeError("Approved interface patch hash differs")
        compiler_source = apply_single_file_diff(
            compiler_source, interface_patch, "test/inductor/test_compiled_autograd.py")
        if digest(compiler_source.encode()) != EXPECTED_APPROVED_COMPILED_SHA:
            raise RuntimeError("Approved compiled-autograd source hash differs")

    installed_builder_source = pathlib.Path(builder_module.__file__).read_text()
    installed_method = method_ast(installed_builder_source)
    if ast_digest(installed_method) != EXPECTED_BUILDER_METHOD_AST_SHA:
        raise RuntimeError("Installed wrap_listlike differs from the qualified baseline method")
    reference_source = args.builder_reference.read_text() if args.builder_reference else installed_builder_source
    reference_method = method_ast(reference_source)
    if ast_digest(reference_method) != EXPECTED_BUILDER_METHOD_AST_SHA:
        raise RuntimeError("Builder reference differs from the qualified baseline method")
    if args.cohort == "boxed":
        candidate_source = apply_single_file_diff(reference_source, patch)
        candidate_method = method_ast(candidate_source)
        if ast_digest(candidate_method) != "291bbefbae2613e645d794f66a961b9b7b33eb3815043fd3c1da6138c4b02f89":
            raise RuntimeError("Patched method differs from the qualified production repair")
        exec(compile(ast.Module(body=[candidate_method], type_ignores=[]),
                     "public-boxed-metadata-control", "exec"), builder_module.__dict__)
        builder_module.VariableBuilder.wrap_listlike = builder_module.wrap_listlike

    default_device = str(torch.get_default_device())
    if torch.get_default_device().type != "cpu":
        raise RuntimeError("The unchanged test requires CPU default device for this CPU control")
    namespace = {
        "__name__": "public_gc_firstcall",
        "__file__": str(args.compiled_autograd_source),
        "torch": torch, "gc": gc, "Function": Function, "TestCase": TestCase,
        "disable_gc": disable_gc, "contextlib": contextlib, "functools": functools,
        "re": re, "unittest": unittest, "compiled_autograd": compiled_autograd,
        "config": config, "inductor": inductor, "aot_eager": aot_eager,
        "counters": counters, "HAS_CUDA_AND_TRITON": HAS_CUDA_AND_TRITON,
        "HAS_XPU_AND_TRITON": HAS_XPU_AND_TRITON, "IS_S390X": IS_S390X,
    }

    test_tree = ast.parse(autograd_source)
    original_class = next(n for n in test_tree.body if isinstance(n, ast.ClassDef) and n.name == "TestAutograd")
    methods = [n for n in original_class.body
               if isinstance(n, ast.FunctionDef) and n.name in (TEST_NAME, "tearDown")]
    if len(methods) != 2:
        raise RuntimeError("Original target method and tearDown are required")
    minimal_class = ast.ClassDef(name=original_class.name, bases=original_class.bases,
                                 keywords=original_class.keywords, body=methods,
                                 decorator_list=original_class.decorator_list)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[minimal_class], type_ignores=[])),
                 str(args.autograd_source), "exec"), namespace)

    compiler_tree = ast.parse(compiler_source)
    functions = {"make_compiler_fn", "make_wrapped", "lookup_backend", "wrap_test_class"}
    assignments = {"known_graph_breaks_tests", "test_contexts", "xfail_re",
                   "xfail_by_backend", "xfail_divergence_from_eager", "skipped_tests"}
    nodes = []
    for node in compiler_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments
                                                for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.If) and any(isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == "skipped_tests"
                for n in ast.walk(node)):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(args.compiled_autograd_source), "exec"), namespace)

    Original = namespace["TestAutograd"]
    original_fn = getattr(Original, TEST_NAME)
    source_backend = namespace["lookup_backend"](TEST_NAME)
    fullgraph = TEST_NAME not in namespace["known_graph_breaks_tests"]
    dynamic = inspect.signature(namespace["make_compiler_fn"]).parameters["dynamic"].default
    if (source_backend, fullgraph, dynamic) != ("inductor", False, True):
        raise RuntimeError("Source-derived target compiler configuration differs")
    if TEST_NAME in namespace["test_contexts"]:
        raise RuntimeError("The target has an unqualified test-specific context")

    if args.mode == "automatic":
        Case = namespace["wrap_test_class"](Original)
        effective_backend = source_backend if getattr(Case, TEST_NAME) is not original_fn else "uncaptured"
        capability_label = "Original automatic availability-gated wrapper selection"
    else:
        effective_backend = source_backend
        compiler = namespace["make_compiler_fn"](backend=effective_backend, fullgraph=fullgraph)
        if args.wrapper_cohort == "approved":
            contexts = [functools.partial(compiled_autograd._enable, compiler), contextlib.nullcontext]
        else:
            contexts = [compiled_autograd._enable(compiler), contextlib.nullcontext()]
        members = Original.__dict__.copy()
        members[TEST_NAME] = namespace["make_wrapped"](original_fn, contexts)
        Case = type("TestAutogradWithCompiledAutograd", (Original,), members)
        Case.__file__ = str(args.compiled_autograd_source)
        capability_label = "Direct CPU compiler-context control; does not claim automatic GPU-gated collection"

    metadata = {
        "cohort": args.cohort, "wrapper_cohort": args.wrapper_cohort,
        "mode": args.mode, "capability_label": capability_label,
        "torch": torch.__version__, "python": __import__("platform").python_version(),
        "has_cuda_and_triton": HAS_CUDA_AND_TRITON, "has_xpu_and_triton": HAS_XPU_AND_TRITON,
        "default_device": default_device, "default_dtype": str(torch.get_default_dtype()),
        "source_backend": source_backend, "effective_backend": effective_backend,
        "fullgraph": fullgraph, "dynamic": dynamic,
        "source_test_specific_context": "nullcontext", "first_call_only": True,
        "original_test_assertions_unchanged": True, "availability_flags_changed": False,
        "installed_source_edits": False,
        "capture_counter_scope": "process_totals_without_reset",
        "capture_counters_before_test": capture_counter_snapshot(counters),
        "source_hashes": {
            "test_autograd": digest(args.autograd_source.read_bytes()),
            "test_compiled_autograd": digest(args.compiled_autograd_source.read_bytes()),
            "active_test_compiled_autograd": digest(compiler_source.encode()),
            "approved_interface_patch": interface_patch_sha,
            "boxed_patch": digest(args.boxed_patch.read_bytes()),
            "installed_builder": digest(installed_builder_source.encode()),
            "baseline_wrap_listlike_ast": ast_digest(installed_method),
        },
    }
    print(json.dumps(metadata), flush=True)
    reference_reports = []
    original_weak_tensor_ref = None
    if args.observe_reference_owners:
        original_weak_tensor_ref = observe_weak_tensor_refs(torch, reference_reports)
    try:
        case = Case(TEST_NAME)
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(unittest.TestSuite([case]))
    finally:
        if original_weak_tensor_ref is not None:
            torch._C._WeakTensorRef = original_weak_tensor_ref
    metadata["capture_counters_after_test"] = capture_counter_snapshot(counters)
    metadata["reference_observer_enabled"] = args.observe_reference_owners
    metadata["reference_owner_reports"] = reference_reports
    metadata.update({
        "tests_run": result.testsRun,
        "failures": [text for _, text in result.failures],
        "errors": [text for _, text in result.errors],
        "skips": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "expected_failure_count": len(result.expectedFailures),
        "unexpected_success_count": len(result.unexpectedSuccesses),
    })
    if result.errors:
        metadata["outcome"] = "ERROR"
    elif result.failures:
        metadata["outcome"] = "FAIL"
    elif result.unexpectedSuccesses:
        metadata["outcome"] = "XPASS"
    elif result.expectedFailures:
        metadata["outcome"] = "XFAIL"
    elif result.skipped:
        metadata["outcome"] = "SKIP"
    else:
        metadata["outcome"] = "PASS"
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"cohort": args.cohort, "mode": args.mode,
                      "outcome": metadata["outcome"]}), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
