"""cortec.help: the help page behind the `cortec` and `cortec-hybrid` commands (and
`python -m cortec`, `python -m cortec_hybrid`).

`--help` is one screen: the run order with a one-line purpose each, the other public names
grouped by what they are, and where to go next. `<command> <name>` prints one function or class:
its arguments one per line with type and default, then its documentation; `<command>
<Class>.<method>` prints one method. Every argument, default and description is read from the
code with `inspect`, so the page cannot disagree with the package it describes.
"""
from __future__ import annotations

import inspect
import textwrap

from .report import REFUSAL_TYPES, WIDTH, emit, header, kv_block, note, paint


# ── reading the code ──────────────────────────────────────────────────────────────────
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


def _first_sentence(obj) -> str:
    text = _first_line(obj)
    head, sep, _ = text.partition(". ")
    return head + "." if sep else text


def _ann(a) -> str:
    if isinstance(a, str):
        return a.strip("'\"")
    return getattr(a, "__name__", None) or repr(a)


def _arguments(obj, *, method: bool = False) -> tuple[list[tuple[str, str]], str]:
    """One (name, description) row per argument, and the return type. `self`/`cls` dropped."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return [("...", "")], ""
    params = list(sig.parameters.values())
    if method and params and params[0].name in ("self", "cls"):
        params = params[1:]
    rows, star = [], False
    for p in params:
        if p.kind == p.VAR_POSITIONAL:
            rows.append(("*" + p.name, "any number of positional values"))
            star = True
            continue
        if p.kind == p.VAR_KEYWORD:
            rows.append(("**" + p.name, "further keyword arguments"))
            continue
        if p.kind == p.KEYWORD_ONLY and not star:
            rows.append(("*", "the arguments below are passed by name"))
            star = True
        desc = _ann(p.annotation) if p.annotation is not p.empty else ""
        if p.default is not p.empty:
            default = "a new empty value" if repr(p.default) == "<factory>" else repr(p.default)
            desc += (", " if desc else "") + "default " + default
        else:
            desc += (", " if desc else "") + "required"
        rows.append((p.name, desc))
    ret = "" if sig.return_annotation is sig.empty else _ann(sig.return_annotation)
    return rows, ret


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
    return inspect.isclass(cls) and any(issubclass(cls, t) for t in REFUSAL_TYPES)


# ── the words in the output ───────────────────────────────────────────────────────────
# The terms the printed records use that are this tool's own, in the order a run prints them.
# Standard vocabulary (differential privacy, Laplace noise, AUC, the three classifiers) is not
# repeated. `cortec terms` prints the list; `cortec <term>` prints one entry.
TERMS_CORTEC = [
    ("epsilon", "the differential-privacy budget; smaller is stronger. `epsilon declared` is what "
                "you asked for and `epsilon accounted` what the ledger charged; the two are equal "
                "when the run is within budget"),
    ("privacy unit", "what one epsilon protects: one row, or one person when `max_rows_per_person` "
                     "is declared; a person who contributes k rows receives k times epsilon"),
    ("n_min", "the smallest cohort or cell that is released; smaller ones are suppressed, because "
              "a rate over fewer records is mostly its own noise"),
    ("cohort", "a group of records formed by a public rule over one column (a band of age, say); "
               "each cohort has its own released histograms"),
    ("conditional table, level, cell", "P(target | cell): the target rate in each cell of a "
                                       "partition of the records by one or more columns. Level 0 "
                                       "is the coarsest partition; the finest level has the most "
                                       "cells"),
    ("support, released size", "the number of private records in a cell or cohort, with noise "
                               "added and charged to the budget"),
    ("class-conditional", "histograms released once per outcome class in a cohort, where both "
                          "classes clear n_min, so the model sees what positives and negatives "
                          "each look like"),
    ("queries, groups, partitions", "every released statistic is a query charged to the ledger. "
                                    "Queries over disjoint parts of the data (partitions) share "
                                    "one charge; the groups add up"),
    ("yield", "rows kept divided by rows requested from the model. Rows outside the schema's "
              "bounds or carrying an undeclared category are dropped and counted, never kept"),
    ("positive rate", "the share of rows whose target is the positive label"),
    ("reasoning evidence", "whether the model reported reasoning tokens on each call. Reasoning is "
                           "the largest single effect measured, and a count that was never "
                           "reported is not a count of zero"),
    ("pool factor", "generate pool_factor times n rows, then keep the n whose cell counts match "
                    "the release; 1 generates exactly n"),
    ("utility transmission bound", "Stage C: for each released cell, a differentially private "
                                   "estimate of the private target rate and a bound on the gap "
                                   "between it and the synthetic rate. The bounds hold over every "
                                   "cell at once with probability 1 - alpha. It bounds utility; it "
                                   "is not a privacy audit"),
    ("tolerance", "the largest per-cell gap you accept (0.15 by default). A table is within "
                  "tolerance when its worst bound is at or below it"),
    ("alpha", "the bound holds over every cell with probability at least 1 - alpha; 0.05 gives "
              "95% simultaneous confidence"),
    ("ceiling", "a real sample of the same size as the synthetic table, scored the same way: the "
                "best any synthetic method can do at that size"),
    ("floor", "the same real sample with its target column permuted: data with no conditional "
              "signal at all"),
    ("discriminating", "the test had power in this run: the ceiling is within tolerance and the "
                       "floor is not. When it is not discriminating, the verdict on the synthetic "
                       "table is not reported, because the test could not have told good from "
                       "bad"),
    ("covered, uncovered", "a released cell is covered when the synthetic table has rows in it. "
                           "An uncovered cell scores the trivial bound 1.0, so a bound cannot be "
                           "earned by covering a convenient subset"),
    ("thin", "a covered cell with fewer than 20 synthetic rows; its bound rests on few rows"),
    ("mean bound, worst bound", "the average and the largest per-cell bound; the verdict uses "
                                "the worst"),
    ("1-way TV", "for each column, the total-variation distance between the table's histogram and "
                 "the real holdout's on the public bins, averaged over the columns. 0 is "
                 "identical; lower is better"),
    ("TSTR-LR, TSTR-RF, TSTR-GBM", "train on synthetic, test on real: a logistic regression, a "
                                   "random forest and a gradient-boosting classifier trained on "
                                   "the table and scored by AUC on the real holdout. Higher is "
                                   "better; read against the ceiling and floor rows"),
    ("absent categories", "declared categories that never appear in the table: a "
                          "representativeness failure that no aggregate measure shows"),
    ("reference rows", "the ceiling and floor rows of the evaluation table, printed in grey. Read "
                       "every other row against them, not against a threshold"),
]


def glossary(terms, *, tool: str, command: str) -> str:
    """The page `<command> terms` prints."""
    lines = [header("Help", "Terms in the output", tool=tool)]
    lines += _para(f"The words the printed records use, in the order a run prints them. Standard "
                   f"vocabulary (differential privacy, Laplace noise, AUC, the classifiers) is not "
                   f"repeated. `{command} <term>` prints one entry.")
    for term, meaning in terms:
        lines += ["", paint("  " + term, "key")] + _para(meaning, 4)
    return "\n".join(lines)


def term_page(terms, name: str, *, tool: str, command: str) -> str | None:
    """The entries whose term contains `name`, or None."""
    key = name.strip().lower()
    hits = [(t, m) for t, m in terms if key in t.lower()]
    if not hits:
        return None
    lines = [header("Help", hits[0][0] if len(hits) == 1 else name, tool=tool)]
    for term, meaning in hits:
        lines += ["", paint("  " + term, "key")] + _para(meaning, 4)
    lines += ["", kv_block([(f"{command} terms", "every term in the output")], indent=2)]
    return "\n".join(lines)


# ── rendering ─────────────────────────────────────────────────────────────────────────
def _para(text: str, indent: int = 2, *, role: str | None = "reference") -> list[str]:
    return [paint(line, role) for line in
            textwrap.wrap(text, WIDTH - 2, initial_indent=" " * indent,
                          subsequent_indent=" " * indent, break_long_words=False,
                          break_on_hyphens=False)]


def _doc_lines(doc: str, indent: int = 2) -> list[str]:
    """A docstring as wrapped paragraphs; indented blocks (code, tables) are kept as they are."""
    lines = []
    for para in doc.split("\n\n"):
        if para.startswith("  ") or para.startswith("\t"):
            lines += [" " * indent + ln for ln in para.splitlines()]
        else:
            lines += _para(para.replace("\n", " "), indent)
        lines.append("")
    return lines


def _labelled_row(label: str, body: str, width: int, *, indent: int = 4,
                  role: str | None = None) -> list[str]:
    """`label` in the key colour, `body` wrapped beside it and aligned under itself."""
    pad = " " * (indent + width + 2)
    wrapped = textwrap.wrap(body, WIDTH - 2, initial_indent=pad, subsequent_indent=pad,
                            break_long_words=False, break_on_hyphens=False) or [pad]
    first = " " * indent + paint(f"{label:<{width}}", "key") + "  " + wrapped[0].lstrip()
    rest = [paint(line, role) for line in wrapped[1:]]
    return [first] + rest


def _names_row(label: str, names: list[str], width: int) -> list[str]:
    return _labelled_row(label, ", ".join(names), width)


def overview(package, *, tool: str, module: str, command: str, run_order, groups, readme: str,
             example: str | None = None, commands=None) -> str:
    """The page `<command> --help` prints: one screen."""
    lines = [header("Help", "What you can run", tool=tool)]
    if commands:
        lines += [paint("  commands", "key"), kv_block(commands, indent=4), ""]
    lines += _para(f"{tool} is a library: each step below is a Python call. `{command} <name>` "
                   f"prints one in full, with its arguments and defaults.")
    lines += ["", paint("  run order", "key"), kv_block(run_order, indent=4)]
    listed = {n for k, _ in run_order for n in k.replace(",", " ").split() if not n[0].isdigit()}
    public = [n for n in package.__all__ if not n.startswith("__")]
    groups = [(label, [n for n in names if n in public]) for label, names in groups]
    for _, names in groups:
        listed.update(names)
    refusals = [n for n in public if n not in listed and _is_refusal(getattr(package, n, None))]
    listed.update(refusals)
    other = [n for n in public if n not in listed and callable(getattr(package, n, None))]
    rows = [(label, names) for label, names in groups if names]
    if refusals:
        rows.append(("refusals", refusals))
    if other:
        rows.append(("other", other))
    if rows:
        width = max(len(label) for label, _ in rows)
        lines += ["", paint("  also public", "key")]
        for label, names in rows:
            lines += _names_row(label, names, width)
        lines.append(note(f"refusals print as a boxed block, not a traceback, under install_guard() "
                          f"or guard()"))
    more = [(f"{command} <name>", "one function or class: arguments, defaults, notes"),
            (f"{command} <Class>.<method>", "one method"),
            (f"{command} terms", "what each word in the output means (tolerance, ceiling ...)"),
            (f"{command} --help", f"this page (or: python -m {module})")]
    if example:
        more.append((f"python {example}", "a runnable tour; no key and no data needed"))
    more.append((readme, "the manual: each step, troubleshooting"))
    lines += ["", paint("  more", "key"), kv_block(more, indent=4)]
    return "\n".join(lines)


def _resolve(package, name: str):
    """A public name, or `Class.method` on a public class; None otherwise."""
    public = [n for n in package.__all__ if not n.startswith("__")]
    head, _, tail = name.partition(".")
    if head not in public:
        return None, False
    obj = getattr(package, head, None)
    if not tail:
        return (obj if callable(obj) else None), False
    if inspect.isclass(obj):
        for mname, fn in _public_methods(obj):
            if mname == tail:
                return fn, True
    return None, False


def detail(package, name: str, *, tool: str, command: str) -> str | None:
    """The page `<command> <name>` prints, or None when the name is not public."""
    obj, is_method = _resolve(package, name)
    if obj is None:
        return None
    lines = [header("Help", name, tool=tool)]
    first = _first_line(obj)
    if first:
        lines += _para(first)
    rows, ret = _arguments(obj, method=is_method)
    if inspect.isclass(obj):
        lines += ["", paint("  to construct", "key")]
    else:
        lines += ["", paint("  arguments", "key")]
    if ret and ret != "None":
        rows.append(("returns", ret))
    lines.append(kv_block(rows, indent=4) if rows else note("  none"))
    rest = _own_doc(obj).split("\n\n", 1)
    if len(rest) > 1 and rest[1].strip():
        lines += ["", paint("  notes", "key")] + _doc_lines(rest[1], 4)
    if inspect.isclass(obj):
        methods = _public_methods(obj)
        if methods:
            width = max(len(m) for m, _ in methods)
            lines += ["", paint("  methods", "key")]
            for m, fn in methods:
                lines += _labelled_row(m, _first_sentence(fn) or "(no description)", width)
            lines.append(note(f"{command} {name}.<method> prints one of them in full"))
    if _is_refusal(obj):
        lines.append(note("a refusal: printed as a boxed block, not a traceback, under install_guard()"))
    lines += ["", kv_block([(f"{command} --help", "the overview of every public name")], indent=2)]
    return "\n".join(lines)


def main(package, argv, *, tool: str, module: str, command: str, run_order, groups, readme: str,
         example: str | None = None, subcommands: dict | None = None, terms=None) -> int:
    """Behind the `cortec` and `cortec-hybrid` commands and `python -m <module>`: a subcommand
    (`run`) is dispatched with the rest of the arguments; `--help`, `-h` or no argument prints
    the overview; `terms` prints the glossary; a public name prints its page, a term its entry;
    an unknown name lists the public names and returns 2."""
    if argv and subcommands and argv[0] in subcommands:
        return int(subcommands[argv[0]](argv[1:]) or 0)
    if not argv or argv[0] in ("-h", "--help", "help"):
        commands = [(f"{command} {name} ...", fn.__doc__ or "") for name, fn in (subcommands or {}).items()]
        emit(overview(package, tool=tool, module=module, command=command, run_order=run_order,
                      groups=groups, readme=readme, example=example, commands=commands))
        return 0
    if argv[0] == "terms" and terms:
        emit(glossary(terms, tool=tool, command=command))
        return 0
    page = detail(package, argv[0], tool=tool, command=command)
    if page is None and terms:
        page = term_page(terms, argv[0], tool=tool, command=command)
    if page is None:
        names = ", ".join(n for n in package.__all__ if not n.startswith("__"))
        emit(header("Help", "Unknown name", tool=tool))
        emit(kv_block([("asked for", argv[0])]))
        emit("")
        for line in _para("public names: " + names, 2):
            emit(line)
        emit(kv_block([(f"{command} --help", "the overview"), (f"{command} terms", "the words in the output")]))
        return 2
    emit(page)
    return 0
