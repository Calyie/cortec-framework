"""cortec.report: one output standard for both tools.

Every run prints the same layout: a stage header, a block of named values, one or more tables,
the warnings, and a verdict. Every run can export the same record: one JSON file with a declared
format version, a Markdown rendering of the same content, one CSV per table, and a figure when
matplotlib is installed (`cortec.plots`). The palette, the type scale and the table conventions
are the paper's, so a table printed here reads like the paper's tables and a figure saved here
looks like the paper's figures. `docs/result-format.md` documents the record.

Terminal colour is used only when stdout is a terminal. `NO_COLOR` disables it, `CORTEC_COLOR=0`
disables it, `CORTEC_COLOR=1` forces it (for a log that will be read with colour). Every function
that returns text returns plain text with colour off, and the tests run with colour off.

Colour roles, fixed across the two tools:

    result     blue    a number this run produced
    reference  grey    a ceiling or floor row, a note, or a value copied from the input
    ok         green   a verdict in the output's favour, or a check that passed
    warning    amber   a condition to read before proceeding
    refusal    orange  a verdict against the output, or the tool declining to proceed
"""
from __future__ import annotations

import csv
import json
import math
import numbers
import os
import re
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

WIDTH = 88
FORMAT = "cortec-result/1"

# The paper's validated categorical palette (paper/make_figures.py), assigned in fixed order, and
# its ink colours. The same hex values drive the terminal colours and the figures.
PALETTE = {"blue": "#2a78d6", "orange": "#d1541f", "aqua": "#0f8f68", "yellow": "#a8770a",
           "magenta": "#c4306b", "violet": "#4a3aa7", "green": "#0f7a3d"}
SERIES = ("blue", "orange", "aqua", "yellow", "magenta", "violet", "green")
INK, INK2, MUTED, GRID = "#111111", "#4e4d4a", "#8a8985", "#e6e5e2"
ROLE_HEX = {"result": PALETTE["blue"], "reference": MUTED, "ok": PALETTE["green"],
            "warning": PALETTE["yellow"], "refusal": PALETTE["orange"], "title": INK, "key": INK2}


# ── colour ────────────────────────────────────────────────────────────────────────────
def colour_enabled(stream=None) -> bool:
    """True when output should carry ANSI colour: forced by CORTEC_COLOR, else a terminal."""
    v = os.environ.get("CORTEC_COLOR")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "always", "on")
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _ansi_fg(hex_: str) -> str:
    r, g, b = _rgb(hex_)
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return f"\x1b[38;2;{r};{g};{b}m"
    q = lambda c: int(round(c / 255 * 5))        # nearest step of the 6x6x6 cube (Terminal.app)
    return f"\x1b[38;5;{16 + 36 * q(r) + 6 * q(g) + q(b)}m"


def paint(text: str, role: str | None = None, *, bold: bool = False,
          enabled: bool | None = None) -> str:
    """`text` in the colour of `role`, or unchanged when colour is off."""
    on = colour_enabled() if enabled is None else enabled
    if not on or (role is None and not bold):
        return text
    codes = ("\x1b[1m" if bold else "") + (_ansi_fg(ROLE_HEX[role]) if role else "")
    return f"{codes}{text}\x1b[0m"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_colour(text: str) -> str:
    return _ANSI.sub("", text)


# ── formatting ────────────────────────────────────────────────────────────────────────
def fmt(v, decimals: int = 3) -> str:
    """One number format for every table and value block: booleans as yes/no, integers with a
    thousands separator, floats to `decimals` places (the paper reports three), missing as n/a."""
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        f = float(v)
        return "n/a" if math.isnan(f) else f"{f:.{decimals}f}"
    return str(v)


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))


def _plain(v):
    """A JSON-safe copy of a value."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v if v is None or isinstance(v, str) else str(v)


# ── blocks ────────────────────────────────────────────────────────────────────────────
def header(stage: str, title: str, *, tool: str = "cortec", version: str | None = None,
           enabled: bool | None = None) -> str:
    """The rule that opens every stage: `── cortec 0.3.0 · Stage A · Release ────`."""
    if version is None:
        version = _package_version(tool)
    left = f"── {tool} {version} · {stage} · {title} "
    return paint(left + "─" * max(0, WIDTH - len(left)), "title", bold=True, enabled=enabled)


def _package_version(tool: str) -> str:
    try:
        if tool == "cortec-hybrid":
            from cortec_hybrid import __version__ as v
        else:
            from cortec import __version__ as v
        return v
    except Exception:
        return ""


def kv_block(pairs, *, indent: int = 2, decimals: int = 3, enabled: bool | None = None) -> str:
    """Named values, keys in grey and numbers in blue, one per line, aligned."""
    pairs = [(str(k), v) for k, v in pairs]
    if not pairs:
        return ""
    w = max(len(k) for k, _ in pairs)
    lines = []
    for k, v in pairs:
        role = "result" if _is_number(v) or isinstance(v, (bool, np.bool_)) else None
        lines.append(" " * indent + paint(f"{k:<{w}}", "key", enabled=enabled) + "  "
                     + paint(fmt(v, decimals), role, enabled=enabled))
    return "\n".join(lines)


def note(text: str, *, enabled: bool | None = None) -> str:
    return paint(f"  {text}", "reference", enabled=enabled)


def warn(text: str, *, enabled: bool | None = None) -> str:
    """The `!!` prefix is the one the package has always printed; colour is added, text is not changed."""
    return paint(f"  !! {text}", "warning", enabled=enabled)


def refuse(text: str, *, enabled: bool | None = None) -> str:
    return paint(f"  REFUSED: {text}", "refusal", enabled=enabled)


def ok(text: str, *, enabled: bool | None = None) -> str:
    return paint(f"  {text}", "ok", enabled=enabled)


def progress(text: str, *, enabled: bool | None = None) -> str:
    return paint(f"    {text}", "reference", enabled=enabled)


_stdout_open = True


def emit(text: str) -> None:
    """Print one line of the record to stdout. Once a downstream reader closes the pipe (`head`,
    quitting `less`), stop silently rather than raising BrokenPipeError on every remaining line."""
    global _stdout_open
    if not _stdout_open:
        return
    try:
        print(text, flush=True)
    except BrokenPipeError:
        _stdout_open = False
        # stdout is redirected to /dev/null so the interpreter's flush at exit does not report
        # the closed pipe a second time
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass


def emit_progress(text: str) -> None:
    """A transient one-line progress indicator on STDERR, redrawn in place after each model call.

    It exists so a long Stage B (dozens of sequential model calls) does not look frozen. It is
    shown only when stderr is a terminal, so it never enters a piped stdout record, a log file or
    an exported record: the `cortec-result/1` contract on stdout is left exactly as before. Pair
    with `emit_progress_done()` to clear the line before the next stdout block."""
    try:
        if not sys.stderr.isatty():
            return
        line = paint(text, "reference", enabled=colour_enabled(sys.stderr))
        sys.stderr.write("\r\x1b[2K  " + line)
        sys.stderr.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass


def emit_progress_done() -> None:
    """Clear the transient progress line so the authoritative stdout summary that follows starts
    clean. The stdout record already carries the final call, row and spend counts."""
    try:
        if sys.stderr.isatty():
            sys.stderr.write("\r\x1b[2K")
            sys.stderr.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass


# ── refusals ──────────────────────────────────────────────────────────────────────────
# Both tools stop on purpose in defined situations: a schema that does not match the data, a
# model outside the validated set, a table no cell of which reaches n_min, a budget that would
# give one person a vacuous guarantee. Each is raised as one of the packages' own exception
# classes, registered here, so that a run script can print it in the standard layout and exit
# instead of showing a traceback. Any other exception is a fault and keeps its traceback.
REFUSAL_TYPES: list[type[BaseException]] = []
REFUSAL_EXIT_CODE = 2


def register_refusal(*types: type[BaseException]) -> None:
    """Declare exception classes as deliberate refusals, rendered by `guard()`."""
    for t in types:
        if t not in REFUSAL_TYPES:
            REFUSAL_TYPES.append(t)


def _tool_of(exc: BaseException) -> str:
    return "cortec-hybrid" if exc.__class__.__module__.startswith("cortec_hybrid") else "cortec"


def render_refusal(exc: BaseException, *, tool: str | None = None,
                   enabled: bool | None = None) -> str:
    """A refusal in the standard layout: the stage header, the class that refused, the reason
    wrapped to the page width, and how to read it. `guard()` prints exactly this."""
    name = exc.__class__.__name__
    title = re.sub(r"(?<!^)(?=[A-Z])", " ", re.sub(r"Error$", "", name)).strip() or name
    lines = [header("Refused", title, tool=tool or _tool_of(exc), enabled=enabled),
             kv_block([("refused by", name)], enabled=enabled),
             paint("  reason", "key", enabled=enabled)]
    for para in str(exc).split("\n"):
        for w in textwrap.wrap(para, WIDTH - 4) or [""]:
            lines.append(paint("    " + w, "refusal", enabled=enabled))
    lines.append("")
    reading = ("The tool stopped on purpose; this is a refusal, not a fault in the tool. "
               "Change the input or the setting the reason names and run again.")
    lines.extend(note(w, enabled=enabled) for w in textwrap.wrap(reading, WIDTH - 2))
    return "\n".join(lines)


@contextmanager
def guard(*, exit_code: int = REFUSAL_EXIT_CODE):
    """Wrap a run so a refusal by either package prints in the standard layout and exits with
    `exit_code`; any other exception keeps its traceback. `with guard(): ...`"""
    try:
        yield
    except tuple(REFUSAL_TYPES) as e:
        emit_progress_done()
        emit(render_refusal(e))
        raise SystemExit(exit_code) from None


def install_guard(*, exit_code: int = REFUSAL_EXIT_CODE) -> None:
    """The same as `guard()` for a whole script: one call at the top installs an exception hook
    that renders a registered refusal and exits, and hands every other exception to the default
    hook unchanged. Interactive shells and notebooks ignore the hook; use `guard()` there."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if isinstance(exc, tuple(REFUSAL_TYPES)):
            emit_progress_done()
            emit(render_refusal(exc))
            sys.exit(exit_code)
        previous(exc_type, exc, tb)

    sys.excepthook = hook


# ── tables ────────────────────────────────────────────────────────────────────────────
@dataclass
class ResultTable:
    """One table in the paper's convention: a label column on the left, numbers right-aligned to
    three places, reference rows (a real sample, a permuted floor) marked so they print in grey."""
    name: str
    columns: list[str]
    rows: list[list]
    reference_rows: list[int] = field(default_factory=list)
    note: str | None = None
    decimals: int = 3

    def _cells(self) -> list[list[str]]:
        return [[fmt(v, self.decimals) for v in r] for r in self.rows]

    def _numeric_columns(self) -> list[bool]:
        out = []
        for j in range(len(self.columns)):
            vals = [r[j] for r in self.rows if j < len(r) and r[j] is not None]
            out.append(bool(vals) and all(_is_number(v) or isinstance(v, (bool, np.bool_)) for v in vals))
        return out

    def render(self, *, enabled: bool | None = None, indent: int = 2) -> str:
        cells = self._cells()
        numeric = self._numeric_columns()
        widths = [max([len(c)] + [len(r[j]) for r in cells if j < len(r)]) for j, c in enumerate(self.columns)]
        pad = " " * indent
        def line(parts, roles):
            out = []
            for j, (p, role) in enumerate(zip(parts, roles)):
                s = f"{p:>{widths[j]}}" if numeric[j] else f"{p:<{widths[j]}}"
                out.append(paint(s, role, enabled=enabled))
            return pad + "  ".join(out)
        lines = [pad + paint(self.name, "title", bold=True, enabled=enabled)]
        lines.append(line(self.columns, ["key"] * len(self.columns)))
        lines.append(pad + paint("─" * (sum(widths) + 2 * (len(widths) - 1)), "reference", enabled=enabled))
        for i, r in enumerate(cells):
            if i in self.reference_rows:
                roles = ["reference"] * len(r)
            else:
                roles = ["result" if numeric[j] and j > 0 else None for j in range(len(r))]
            lines.append(line(r, roles))
        if self.note:
            lines.append(note(self.note, enabled=enabled))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        numeric = self._numeric_columns()
        head = "| " + " | ".join(self.columns) + " |"
        rule = "|" + "|".join("---:" if n else "---" for n in numeric) + "|"
        body = []
        for i, r in enumerate(self._cells()):
            cells = [f"*{c}*" if i in self.reference_rows else c for c in r]
            body.append("| " + " | ".join(cells) + " |")
        out = [f"**{self.name}**", "", head, rule] + body
        if self.note:
            out += ["", self.note]
        return "\n".join(out)

    def to_csv(self, path: str) -> str:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.columns + ["reference"])
            for i, r in enumerate(self.rows):
                w.writerow([_plain(v) for v in r] + [i in self.reference_rows])
        return path

    def to_dict(self) -> dict:
        return {"name": self.name, "columns": list(self.columns),
                "rows": [[_plain(v) for v in r] for r in self.rows],
                "reference_rows": list(self.reference_rows), "note": self.note,
                "decimals": self.decimals}

    @classmethod
    def from_dict(cls, d: dict) -> "ResultTable":
        return cls(name=d["name"], columns=list(d["columns"]), rows=[list(r) for r in d["rows"]],
                   reference_rows=list(d.get("reference_rows", [])), note=d.get("note"),
                   decimals=int(d.get("decimals", 3)))


# ── the record ────────────────────────────────────────────────────────────────────────
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@dataclass
class RunRecord:
    """The exportable result of one run of either tool. `docs/result-format.md` is the contract."""
    tool: str                       # "cortec" or "cortec-hybrid"
    stage: str                      # "Stage A", "Stage B", "Stage C", "Evaluation", "Correction"
    title: str                      # "Release", "Generate", "Utility transmission bound", ...
    schema: str
    values: list = field(default_factory=list)      # [(name, value), ...] in display order
    tables: list = field(default_factory=list)      # [ResultTable, ...]
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    verdict: str | None = None
    verdict_role: str = "ok"                        # ok | refusal | warning
    epsilon: dict = field(default_factory=dict)     # declared, accounted, per_row, per_person
    files: dict = field(default_factory=dict)       # name -> path, filled by save()
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    version: str = ""
    format: str = FORMAT

    def __post_init__(self):
        if not self.version:
            self.version = _package_version(self.tool)

    # -- lookups ---------------------------------------------------------------------
    def value(self, name: str, default=None):
        for k, v in self.values:
            if k == name:
                return v
        return default

    def table(self, name: str) -> ResultTable | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    # -- terminal ----------------------------------------------------------------------
    def render(self, *, enabled: bool | None = None) -> str:
        parts = [header(self.stage, self.title, tool=self.tool, version=self.version, enabled=enabled)]
        if self.values:
            parts.append(kv_block([("schema", self.schema)] + list(self.values), enabled=enabled))
        for t in self.tables:
            parts.append("")
            parts.append(t.render(enabled=enabled))
        if self.warnings:
            parts.append("")
            parts.extend(warn(w, enabled=enabled) for w in self.warnings)
        if self.verdict:
            parts.append("")
            role = {"ok": "ok", "refusal": "refusal", "warning": "warning"}[self.verdict_role]
            parts.append(paint(f"  {self.verdict}", role, bold=True, enabled=enabled))
        if self.notes:
            parts.append("")
            parts.extend(note(n, enabled=enabled) for n in self.notes)
        return "\n".join(parts)

    def show(self) -> None:
        emit(self.render())

    # -- exports -----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"format": self.format, "tool": self.tool, "version": self.version,
                "stage": self.stage, "title": self.title, "schema": self.schema,
                "created": self.created,
                "values": {k: _plain(v) for k, v in self.values},
                "value_order": [k for k, _ in self.values],
                "epsilon": _plain(self.epsilon),
                "tables": [t.to_dict() for t in self.tables],
                "warnings": list(self.warnings), "notes": list(self.notes),
                "verdict": self.verdict, "verdict_role": self.verdict_role,
                "files": dict(self.files)}

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} record: format={d.get('format')!r}")
        order = d.get("value_order") or list(d.get("values", {}))
        return cls(tool=d["tool"], stage=d["stage"], title=d["title"], schema=d["schema"],
                   values=[(k, d["values"][k]) for k in order],
                   tables=[ResultTable.from_dict(t) for t in d.get("tables", [])],
                   warnings=list(d.get("warnings", [])), notes=list(d.get("notes", [])),
                   verdict=d.get("verdict"), verdict_role=d.get("verdict_role", "ok"),
                   epsilon=dict(d.get("epsilon", {})), files=dict(d.get("files", {})),
                   created=d.get("created", ""), version=d.get("version", ""))

    @classmethod
    def from_json(cls, path: str) -> "RunRecord":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_markdown(self) -> str:
        out = [f"## {self.tool} {self.version}: {self.stage}, {self.title}", "",
               f"Schema `{self.schema}`, recorded {self.created}.", ""]
        if self.values:
            out += ["| value | |", "|---|---:|"]
            out += [f"| {k} | {fmt(v)} |" for k, v in self.values]
            out.append("")
        for t in self.tables:
            out += [t.to_markdown(), ""]
        if self.warnings:
            out += ["**Warnings**", ""] + [f"- {w}" for w in self.warnings] + [""]
        if self.verdict:
            out += [f"**Verdict.** {self.verdict}", ""]
        if self.notes:
            out += [f"*{n}*" for n in self.notes] + [""]
        return "\n".join(out).rstrip() + "\n"

    def write_markdown(self, path: str) -> str:
        with open(path, "w") as f:
            f.write(self.to_markdown())
        return path

    def write_csv(self, stem: str) -> dict:
        """`<stem>.csv` holds the named values; each table goes to `<stem>_<table>.csv`."""
        files = {}
        path = f"{stem}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "value"])
            for k, v in [("tool", self.tool), ("stage", self.stage), ("schema", self.schema)] + list(self.values):
                w.writerow([k, _plain(v)])
        files["values_csv"] = path
        for t in self.tables:
            p = f"{stem}_{_slug(t.name)}.csv"
            t.to_csv(p)
            files[f"table_csv:{t.name}"] = p
        return files

    def save(self, directory: str, stem: str | None = None, *, figure: bool = True) -> dict:
        """Write every export under `directory` and return {name: path}. The figure is written
        when matplotlib is installed and the record has one; otherwise it is skipped, and the
        record says so in `files`."""
        os.makedirs(directory, exist_ok=True)
        stem = stem or _slug(f"{self.stage}_{self.title}")
        base = os.path.join(directory, stem)
        files = {"json": f"{base}.json", "markdown": f"{base}.md"}
        files.update(self.write_csv(base))
        if figure:
            try:
                from . import plots
                p = plots.figure_for(self, f"{base}.png")
                if p:
                    files["figure"] = p
            except ImportError as e:
                files["figure"] = f"not written: {e}"
        self.files = files
        self.write_markdown(files["markdown"])
        self.to_json(files["json"])
        return files


# ── builders: one per tool object ────────────────────────────────────────────────────
def record_release(release, *, n_private_rows: int | None = None) -> RunRecord:
    """Stage A: the release's accounting, its cohorts and the finest conditional level."""
    a = release.audit or {}
    k = a.get("max_rows_per_person", "UNDECLARED")
    per_person = a.get("epsilon_per_person")
    levels = release.conditional_levels or []
    finest = levels[-1] if levels else {"columns": [], "cells": {}}
    values = [("epsilon declared", a.get("epsilon_declared", release.epsilon_total)),
              ("epsilon accounted", a.get("epsilon_accounted")),
              ("within budget", a.get("within_budget")),
              ("privacy unit", "undeclared" if k == "UNDECLARED" else f"{k} row(s) per person"),
              ("epsilon per person", per_person if _is_number(per_person) else "undeclared"),
              ("n_min", release.n_min),
              ("queries", a.get("n_queries")),
              ("cohorts", release.n_cohorts),
              ("conditional levels", len(levels)),
              ("cells at the finest level", len(finest.get("cells", {}))),
              ("finest level columns", ", ".join(finest.get("columns", [])) or "none")]
    if n_private_rows is not None:
        values.insert(0, ("private rows read", n_private_rows))
    tables = []
    pos = None
    if release.cohorts:
        rows = []
        for c in release.cohorts:
            cb = c.get("class_balance", {})
            pos = max(cb.values()) if cb else None
            keys = list(cb)
            rows.append([c["cohort_name"], c["cohort_size"], cb.get(keys[0]) if keys else None,
                         "yes" if c.get("by_class") else "no"])
        tables.append(ResultTable("cohorts", ["cohort", "released size", "positive rate", "class-conditional"],
                                  rows, note="sizes and rates are noised; the positive rate is the first declared label's"))
    if finest.get("cells"):
        rows = [[cell, d["rate"], d["support"]] for cell, d in finest["cells"].items()]
        tables.append(ResultTable(f"conditional table, level {finest.get('level', len(levels) - 1)}",
                                  ["cell", "rate", "support"], rows,
                                  note="rate is P(positive | cell) with Laplace noise; support is the noised cell size"))
    groups = a.get("groups", {})
    if groups:
        rows = [[g, d.get("n_queries"), d.get("n_partitions"), d.get("group_total")] for g, d in groups.items()]
        tables.append(ResultTable("privacy accounting", ["group", "queries", "partitions", "epsilon"], rows,
                                  note="groups compose sequentially; partitions within a group in parallel"))
    warnings = list(a.get("warnings", []))
    rec = RunRecord(tool="cortec", stage="Stage A", title="Release", schema=release.schema_name,
                    values=values, tables=tables, warnings=warnings,
                    epsilon={"declared": a.get("epsilon_declared", release.epsilon_total),
                             "accounted": a.get("epsilon_accounted"),
                             "per_row": a.get("epsilon_accounted"),
                             "per_person": per_person if _is_number(per_person) else None},
                    notes=["This is an accounting record, not a privacy audit. Generation from this "
                           "release is post-processing and spends no further budget."])
    if a.get("within_budget") is False:
        rec.verdict, rec.verdict_role = "the release exceeded its declared budget", "refusal"
    return rec


def record_generation(stats, *, backend: str | None = None, surface: str | None = None,
                      model: str | None = None, n_rows: int | None = None,
                      positive_rate: float | None = None, schema: str = "") -> RunRecord:
    """Stage B: the run's counts, the reasoning evidence and the spend."""
    values = []
    if backend:
        values.append(("backend", backend if not model else f"{backend} / {model}"))
    if surface:
        values.append(("surface", surface))
    values += [("calls", stats.calls), ("calls parsed", stats.parse_ok),
               ("rows requested", stats.rows_requested), ("rows kept", stats.rows),
               ("yield", stats.yield_rate),
               ("rows dropped, out of bounds", stats.rows_out_of_bounds),
               ("rows dropped, undeclared category", stats.rows_undeclared_category),
               ("rate-limit waits", stats.rate_limit_waits),
               ("spend (USD)", stats.spend_usd)]
    if getattr(stats, "budget_capped", False):
        values.append(("budget capped", True))
    if n_rows is not None:
        values.append(("rows in the output", n_rows))
    if positive_rate is not None:
        values.append(("positive rate in the output", positive_rate))
    reasoning = ResultTable("reasoning evidence",
                            ["measure", "count"],
                            [["reasoning tokens reported", stats.thinking_tokens],
                             ["calls with a reasoning block", stats.calls_with_reasoning_block],
                             ["calls that reported zero reasoning", stats.calls_without_reasoning],
                             ["calls with no reasoning count at all", stats.calls_reasoning_unmeasured]],
                            note="a count that was never reported is not a count of zero")
    rec = RunRecord(tool="cortec", stage="Stage B", title="Generate", schema=schema, values=values,
                    tables=[reasoning], warnings=list(stats.warnings),
                    epsilon={"per_row": 0.0},
                    notes=["Generation reads only the release: post-processing, epsilon 0."])
    return rec


def record_bound(report, *, schema: str = "") -> RunRecord:
    """Stage C: the three conditions, the verdict or the refusal, and the accounting."""
    c = report.dp_claim
    rows, ref = [], []
    for i, (lab, r) in enumerate((("synthetic", report.synthetic),
                                  ("real sample (ceiling)", report.ceiling),
                                  ("permuted target (floor)", report.floor))):
        rows.append([lab, r.n_cells, r.n_cells_covered, r.n_cells_uncovered, r.n_cells_thin,
                     r.mean_bound, r.worst_case_bound, bool(r.within_bound)])
        if i > 0:
            ref.append(i)
    t = ResultTable("bound per condition",
                    ["condition", "cells", "covered", "uncovered", "thin", "mean bound",
                     "worst bound", "within tolerance"], rows, reference_rows=ref,
                    note=f"tolerance {report.tolerance}; the bound holds over every cell with "
                         f"probability at least {1 - report.synthetic.alpha:.0%}")
    values = [("level", report.synthetic.level),
              ("columns", ", ".join(report.synthetic.columns) or "global"),
              ("epsilon for the bound", c.get("epsilon_transmission_bound")),
              ("alpha", report.synthetic.alpha), ("tolerance", report.tolerance),
              ("discriminating", report.discriminating)]
    if report.discriminating:
        verdict = (f"synthetic data is {report.verdict} at tolerance {report.tolerance} with "
                   f"simultaneous confidence {1 - report.synthetic.alpha:.0%} over "
                   f"{report.synthetic.n_cells} released cells")
        role = "ok" if report.verdict == "within bound" else "refusal"
    else:
        verdict = ("no verdict: the test did not discriminate. A bound is meaningful only when the "
                   "real-sample ceiling clears the tolerance and the permuted-target floor does not. "
                   "Adjust the tolerance or epsilon_cert; do not report the synthetic result from this run.")
        role = "warning"
    warnings = []
    if c.get("epsilon_per_person_vacuous"):
        warnings.append("the per-person epsilon is vacuous; no per-person privacy claim may be made from this report")
    eps = {"release": c.get("epsilon_release"), "bound": c.get("epsilon_transmission_bound"),
           "per_row": c.get("epsilon_total_per_row"), "per_person": c.get("epsilon_per_person")}
    values += [("epsilon per row (release + bound)", eps["per_row"]),
               ("epsilon per person", eps["per_person"] if _is_number(eps["per_person"]) else "undeclared")]
    return RunRecord(tool="cortec", stage="Stage C", title="Utility transmission bound",
                     schema=schema or report.meta.get("schema", ""), values=values, tables=[t],
                     warnings=warnings, verdict=verdict, verdict_role=role, epsilon=eps,
                     notes=["This bounds utility. It is not a privacy audit."])


def record_correction(gain: dict, table, *, schema: str = "", n_rows: int | None = None,
                      positive_rate_before: float | None = None,
                      positive_rate_after: float | None = None) -> RunRecord:
    """cortec-hybrid: what the correction changed, with the calibration verdict and what it does
    not predict."""
    a = table.audit or {}
    values = [("cells", ", ".join(table.columns)), ("released cells", len(table)),
              ("epsilon for the table", table.epsilon),
              ("epsilon accounted", a.get("epsilon_accounted")),
              ("conditional error before", gain["conditional_error_before"]),
              ("conditional error after", gain["conditional_error_after"]),
              ("reduction", gain["conditional_error_reduction"]),
              ("feature marginal shift", gain["feature_marginal_shift"]),
              ("cells corrected", gain["n_cells_corrected"]),
              ("worth it (calibration only)", gain["worth_it"])]
    if n_rows is not None:
        values.append(("rows", n_rows))
    if positive_rate_before is not None and positive_rate_after is not None:
        values += [("positive rate before", positive_rate_before), ("positive rate after", positive_rate_after)]
    rows = [[cell, table.cells[cell], table.support.get(cell)] for cell in table.cells]
    t = ResultTable("released conditional table", ["cell", "rate", "support"], rows,
                    note="rate is P(positive | cell) with Laplace noise; support is the noised cell size")
    warnings = list(a.get("warnings", []))
    return RunRecord(tool="cortec-hybrid", stage="Correction", title="Relabel the target column",
                     schema=schema, values=values, tables=[t], warnings=warnings,
                     verdict=gain["verdict"], verdict_role="ok" if gain["worth_it"] else "warning",
                     epsilon={"table": table.epsilon, "accounted": a.get("epsilon_accounted"),
                              "per_row": a.get("epsilon_accounted")},
                     notes=["worth_it measures " + gain.get("worth_it_measures", "conditional calibration") + ".",
                            "It does not predict " + gain.get("worth_it_does_not_predict", "downstream utility.")])


def record(obj, **kw) -> RunRecord:
    """Build the record for any tool object: a Release, a GenerationStats, a BoundReport, or the
    (corrected, table, gain) triple that cortec-hybrid's correct() returns."""
    if isinstance(obj, RunRecord):
        return obj
    name = type(obj).__name__
    if name == "Release":
        return record_release(obj, **kw)
    if name == "GenerationStats":
        return record_generation(obj, **kw)
    if name == "Generator":
        return record_generation(obj.stats, backend=getattr(obj, "backend", None),
                                 model=getattr(obj, "model", None),
                                 surface=getattr(obj, "surface", None), schema=obj.schema.name, **kw)
    if name == "BoundReport":
        return record_bound(obj, **kw)
    if isinstance(obj, tuple) and len(obj) == 3 and isinstance(obj[2], dict) and "worth_it" in obj[2]:
        return record_correction(obj[2], obj[1], **kw)
    raise TypeError(f"no record builder for {name}")


def show(obj, **kw) -> RunRecord:
    """Print the standard rendering of a tool object and return its record."""
    rec = record(obj, **kw)
    rec.show()
    return rec
