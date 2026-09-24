"""cortec.help: the help page both tools print, from `python -m cortec` and
`python -m cortec_hybrid`.

Every signature and every description on the page is read from the code with `inspect`, so the
page cannot disagree with the package it describes. The overview lists the run order, then each
public function and class with its arguments and defaults, then the refusals `guard()` prints.
`python -m <package> <name>` prints the full documentation of one public name.
"""
from __future__ import annotations

import inspect
import textwrap

from .report import REFUSAL_TYPES, WIDTH, emit, header, kv_block, note, paint


def _own_doc(obj) -> str:
    """The object's own docstring, dedented; never one inherited from a base class, and never
    the signature line a dataclass writes for itself."""
    doc = obj.__dict__.get("__doc__") if inspect.isclass(obj) else getattr(obj, "__doc__", None)
    doc = inspect.cleandoc(doc) if doc else ""
    if inspect.isclass(obj) and doc.startswith(obj.__name__ + "("):
        return ""
    return doc


def _first_line(obj) -> str:
    return _own_doc(obj).split("\n\n", 1)[0].replace("\n", " ").strip()


def _ann(a) -> str:
    if isinstance(a, str):
        return a.strip("'\"")
    return getattr(a, "__name__", None) or repr(a)


def _signature(obj, *, method: bool = False) -> str:
    """The signature as a user writes the call: annotations unquoted, `self`/`cls` dropped."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    params = list(sig.parameters.values())
    if method and params and params[0].name in ("self", "cls"):
        params = params[1:]
    parts, star = [], False
    for p in params:
        if p.kind == p.VAR_POSITIONAL:
            parts.append("*" + p.name)
            star = True
            continue
        if p.kind == p.VAR_KEYWORD:
            parts.append("**" + p.name)
            continue
        if p.kind == p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        s = p.name
        if p.annotation is not p.empty:
            s += ": " + _ann(p.annotation)
        if p.default is not p.empty:
            s += (" = " if p.annotation is not p.empty else "=") + repr(p.default)
        parts.append(s)
    ret = "" if sig.return_annotation is sig.empty else " -> " + _ann(sig.return_annotation)
    return "(" + ", ".join(parts) + ")" + ret


def _wrapped(text: str, indent: int, *, role: str | None = None, hang: int = 4) -> list[str]:
    lines = textwrap.wrap(text, WIDTH - 2, initial_indent=" " * indent,
                          subsequent_indent=" " * (indent + hang),
                          break_long_words=False, break_on_hyphens=False) or [" " * indent]
    return [paint(line, role) for line in lines]


def _public_methods(cls) -> list[tuple[str, object]]:
    """The methods this class defines itself (not inherited), in definition order."""
    out = []
    for name, obj in vars(cls).items():
        if name.startswith("_"):
            continue
        fn = obj.__func__ if isinstance(obj, (classmethod, staticmethod)) else obj
        if inspect.isfunction(fn):
            out.append((name, fn))
    return out


def _is_refusal(cls) -> bool:
    return any(issubclass(cls, t) for t in REFUSAL_TYPES)


def _grouped(package):
    functions, classes, errors = [], [], []
    for name in package.__all__:
        if name.startswith("__"):
            continue
        obj = getattr(package, name, None)
        if obj is None:
            continue
        if inspect.isclass(obj) and issubclass(obj, BaseException):
            errors.append((name, obj))
        elif inspect.isclass(obj):
            classes.append((name, obj))
        elif callable(obj):
            functions.append((name, obj))
    return functions, classes, errors


def overview(package, *, tool: str, module: str, command: str, run_order, readme: str,
             example: str | None = None) -> str:
    """The page `<command> --help` (and `python -m <module>`) prints."""
    lines = [header("Help", "What you can run", tool=tool), ""]
    lines.append(paint("  run order", "key"))
    lines.append(kv_block(run_order, indent=4))
    functions, classes, errors = _grouped(package)
    if functions:
        lines += ["", paint("  functions", "key")]
        for name, obj in functions:
            lines += _wrapped(name + _signature(obj), 4)
            lines += _wrapped(_first_line(obj), 6, role="reference", hang=0)
    if classes:
        lines += ["", paint("  classes", "key")]
        for name, cls in classes:
            lines += _wrapped(name + _signature(cls), 4)
            first = _first_line(cls)
            if first:
                lines += _wrapped(first, 6, role="reference", hang=0)
            for mname, fn in _public_methods(cls):
                lines += _wrapped("." + mname + _signature(fn, method=True), 6)
    if errors:
        lines += ["", paint("  refusals, printed in this layout by guard() and install_guard()", "key")]
        for name, cls in errors:
            first = _first_line(cls)
            tag = "" if _is_refusal(cls) else "  (a fault, not a refusal: keeps its traceback)"
            lines += _wrapped(name + ("  " + first if first else "") + tag, 4, role="reference")
    more = [(f"{command} <name>", "the full documentation of one name above"),
            (f"{command} --help", f"this page (or: python -m {module})")]
    if example:
        more.append((f"python {example}", "a runnable tour; no key and no data needed"))
    more.append((readme, "the manual: each step, its arguments, what to read"))
    lines += ["", paint("  more", "key"), kv_block(more, indent=4)]
    return "\n".join(lines)


def detail(package, name: str, *, tool: str, command: str) -> str | None:
    """The page `<command> <name>` prints, or None when the name is not public."""
    obj = getattr(package, name, None) if name in getattr(package, "__all__", ()) else None
    if obj is None or not callable(obj):
        return None
    lines = [header("Help", name, tool=tool), ""]
    lines += _wrapped(name + _signature(obj), 2)
    lines.append("")
    for para in (_own_doc(obj) or "(no documentation)").split("\n\n"):
        if para.startswith("  ") or para.startswith("\t"):       # a code or table block: keep as is
            lines += ["  " + ln for ln in para.splitlines()]
        else:
            lines += _wrapped(para.replace("\n", " "), 2, role="reference", hang=0)
        lines.append("")
    if inspect.isclass(obj):
        for mname, fn in _public_methods(obj):
            lines += _wrapped("." + mname + _signature(fn, method=True), 2)
            first = _first_line(fn)
            if first:
                lines += _wrapped(first, 4, role="reference", hang=0)
            lines.append("")
    lines.append(kv_block([(f"{command} --help", "the overview of every public name")]))
    return "\n".join(lines)


def main(package, argv, *, tool: str, module: str, command: str, run_order, readme: str,
         example: str | None = None) -> int:
    """Behind the `cortec` and `cortec-hybrid` commands and `python -m <module>`: `--help`, `-h`
    or no argument prints the overview; a public name prints its page; an unknown name lists
    the public names and returns 2."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        emit(overview(package, tool=tool, module=module, command=command, run_order=run_order,
                      readme=readme, example=example))
        return 0
    page = detail(package, argv[0], tool=tool, command=command)
    if page is None:
        names = ", ".join(n for n in package.__all__ if not n.startswith("__"))
        emit(header("Help", "Unknown name", tool=tool))
        emit(kv_block([("asked for", argv[0])]))
        emit("")
        for line in _wrapped("public names: " + names, 2, role="reference", hang=2):
            emit(line)
        emit(kv_block([(f"{command} --help", "the overview")]))
        return 2
    emit(page)
    return 0
