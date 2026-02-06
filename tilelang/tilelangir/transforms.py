# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
TileLangIR transforms: single-pass functions (str -> str or module -> module).

Tilelangir only exposes run_pass_pipeline(mlir_str, pipeline_str) and does not
provide Context/Module, so lower uses string mode: pass in MLIR string, each
pass returns a new string. Each pass accepts both str and Module for callers
that have a Module.

Pass order is decided by the caller (e.g. lower.py).

Pass options: each pass accepts optional **options; they are forwarded to the MLIR
pipeline string as pass-name{key=value,...}. Keys are converted from snake_case to
kebab-case. E.g. canonicalize(x, top_down=True) -> canonicalize{top-down=true}.

Example (string mode, with options):
    mlir_str = transforms.canonicalize(mlir_str, top_down=True)
"""

# Pass names for reference (order decided in lower.py).
TILELANGIR_COMPILE_PIPELINE_PASSES = [
    "canonicalize",
    "cse",
    "sccp",
    "tilelangir-cv-split",
    "tilelangir-vectorize",
]


def _run_single_pass_on_module(module, pass_spec: str):
    """Run one pass on module via tilelangir native; returns new Module."""
    s = _run_single_pass_str(str(module), pass_spec)
    ctx = getattr(module, "context", None)
    if ctx is None:
        raise RuntimeError("module has no context attribute")
    parse = type(module).parse
    try:
        return parse(s, ctx)
    except TypeError:
        return parse(ctx, s)


def _pass_str_or_module(x, pass_spec: str):
    """Run one pass: if x is str return str; if x is a Module return new Module."""
    if isinstance(x, str):
        return _run_single_pass_str(x, pass_spec)
    return _run_single_pass_on_module(x, pass_spec)


def _format_pass_option_value(value) -> str:
    """Format a Python value for MLIR pipeline option (e.g. bool -> 'true'/'false')."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _pass_spec(pass_name: str, **options) -> str:
    """Build pipeline spec for one pass: builtin.module(pass-name) or builtin.module(pass-name{k=v,...})."""
    if not options:
        return f"builtin.module({pass_name})"
    opts = ",".join(
        f"{k.replace('_', '-')}={_format_pass_option_value(v)}" for k, v in options.items()
    )
    return f"builtin.module({pass_name}{{{opts}}})"


def canonicalize(x, **options):
    return _pass_str_or_module(x, _pass_spec("canonicalize", **options))


def cse(x, **options):
    return _pass_str_or_module(x, _pass_spec("cse", **options))


def sccp(x, **options):
    return _pass_str_or_module(x, _pass_spec("sccp", **options))


def cv_split(x, **options):
    return _pass_str_or_module(x, _pass_spec("tilelangir-cv-split", **options))


def vectorize(x, **options):
    return _pass_str_or_module(x, _pass_spec("tilelangir-vectorize", **options))


def _run_single_pass_str(mlir_str: str, pass_spec: str) -> str:
    """Internal: run one pass on MLIR string via tilelangir native (e.g. pass_spec='builtin.module(canonicalize)')."""
    try:
        import tilelang.tilelangir as _tilelangir  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TileLangIR passes require tilelangir native module (libtilelangir). "
            "Build with USE_NPUIR and set PYTHONPATH to tilelangir build dir."
        ) from e
    _native = getattr(_tilelangir, "_native", None)
    if _native is None or not hasattr(_native, "run_pass_pipeline"):
        raise RuntimeError(
            "tilelangir native module or run_pass_pipeline not found; rebuild tilelangir Python extension."
        )
    ok, result_or_err = _native.run_pass_pipeline(mlir_str, pass_spec)
    if not ok:
        raise RuntimeError(result_or_err)
    return result_or_err
