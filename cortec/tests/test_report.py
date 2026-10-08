"""The output standard: every run prints the same layout and exports the same record.

Each test is named after what it prevents: colour codes reaching a log or a test, a number
formatted two ways in one table, a record that cannot be read back, an export that silently
skips a file, a figure that does not match the paper's style, and an evaluation that scores a
table against a threshold instead of beside a real sample and a permuted floor.
"""
import csv
import json
import os

import numpy as np
import pandas as pd
import pytest

from cortec import report
from cortec.report import (ResultTable, RunRecord, fmt, fmt_seconds, paint, strip_colour, record,
                           record_release, record_generation, record_bound, record_results)


@pytest.fixture(autouse=True)
def _colour_off(monkeypatch):
    monkeypatch.setenv("CORTEC_COLOR", "0")


# ── colour ────────────────────────────────────────────────────────────────────────────
def test_colour_is_off_unless_forced_or_a_terminal(monkeypatch):
    assert paint("x", "result") == "x"
    assert report.warn("careful") == "  !! careful"          # the prefix the package always printed
    monkeypatch.setenv("CORTEC_COLOR", "1")
    painted = paint("x", "result", bold=True)
    assert "\x1b[" in painted and strip_colour(painted) == "x"
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert "38;2;42;120;214" in paint("x", "result")        # the paper's blue, exactly
    monkeypatch.delenv("CORTEC_COLOR")
    monkeypatch.setenv("NO_COLOR", "1")
    assert paint("x", "result") == "x"


def test_every_role_has_a_colour_from_the_paper_palette():
    for role, hex_ in report.ROLE_HEX.items():
        assert hex_ in set(report.PALETTE.values()) | {report.INK, report.INK2, report.MUTED}, role
    assert report.ROLE_HEX["result"] == report.PALETTE["blue"]


# ── formatting ────────────────────────────────────────────────────────────────────────
def test_one_number_format_for_every_table():
    assert fmt(True) == "yes" and fmt(np.bool_(False)) == "no"
    assert fmt(12345) == "12,345" and fmt(np.int64(3)) == "3"
    assert fmt(0.12345) == "0.123" and fmt(np.float64(0.5)) == "0.500" and fmt(2.0, 1) == "2.0"
    assert fmt(float("nan")) == "n/a" and fmt(None) == "n/a"
    assert fmt("text") == "text"


def test_table_right_aligns_numbers_and_marks_reference_rows():
    t = ResultTable("scores", ["table", "rows", "AUC"],
                    [["synthetic", 300, 0.71234], ["real sample", 300, 0.7654], ["floor", 300, 0.5]],
                    reference_rows=[1, 2], note="read against the references")
    lines = t.render().split("\n")
    assert lines[0].strip() == "scores"
    body = lines[3:6]
    # the numeric columns end at the same position on every row
    assert len({len(l) for l in body}) == 1
    assert body[0].endswith("0.712") and body[2].endswith("0.500")
    assert lines[-1].strip() == "read against the references"
    md = t.to_markdown()
    assert "|---|---:|---:|" in md and "| *real sample* | *300* | *0.765* |" in md


def test_table_csv_keeps_full_precision_and_the_reference_flag(tmp_path):
    t = ResultTable("s", ["table", "AUC"], [["a", 0.123456789], ["b", 0.5]], reference_rows=[1])
    p = t.to_csv(str(tmp_path / "s.csv"))
    rows = list(csv.DictReader(open(p)))
    assert rows[0]["AUC"] == "0.123456789" and rows[0]["reference"] == "False"
    assert rows[1]["reference"] == "True"


# ── the record ────────────────────────────────────────────────────────────────────────
def _record():
    t = ResultTable("fidelity and utility", ["table", "rows", "1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM"],
                    [["synthetic", 300, 0.04, 0.70, 0.71, 0.72], ["real", 300, 0.03, 0.76, 0.72, 0.72],
                     ["floor", 300, 0.03, 0.50, 0.55, 0.50]], reference_rows=[1, 2])
    return RunRecord(tool="cortec", stage="Evaluation", title="Fidelity and utility beside real references",
                     schema="toy", values=[("tables scored", 1), ("reference size n", 300), ("note", "x")],
                     tables=[t], warnings=["one warning"], verdict="fine", verdict_role="ok",
                     epsilon={"per_row": 2.0}, notes=["a note"])


def test_record_round_trips_through_json(tmp_path):
    r = _record()
    p = r.to_json(str(tmp_path / "r.json"))
    back = RunRecord.from_json(p)
    assert back.to_dict() == r.to_dict()
    assert back.value("reference size n") == 300 and back.table("fidelity and utility").rows[1][3] == 0.76
    d = json.load(open(p))
    assert d["format"] == report.FORMAT and d["value_order"] == ["tables scored", "reference size n", "note"]


def test_record_refuses_a_file_of_another_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": "something-else/9", "tool": "cortec"}))
    with pytest.raises(ValueError):
        RunRecord.from_json(str(p))


def test_render_has_header_values_table_warning_verdict_in_that_order(capsys):
    out = _record().render()
    parts = ["── CoRTeC", "Evaluation", "schema", "toy", "fidelity and utility", "!! one warning", "fine", "a note"]
    positions = [out.find(p) for p in parts]
    assert all(p >= 0 for p in positions) and positions == sorted(positions)
    assert "\x1b[" not in out
    # a deliberate refusal prints in the same layout and exits, instead of a traceback
    from cortec import DataValidationError
    from cortec.report import guard
    with pytest.raises(SystemExit) as e, guard():
        raise DataValidationError("column 'x' is missing. NO privacy budget was spent.")
    assert e.value.code == 2
    out = capsys.readouterr().out
    parts = ["── CoRTeC", "Refused · Data Validation", "refused by", "DataValidationError",
             "reason", "NO privacy budget was spent", "not a fault in the tool"]
    positions = [out.find(p) for p in parts]
    assert all(p >= 0 for p in positions) and positions == sorted(positions)
    # anything that is not a registered refusal keeps its traceback
    with pytest.raises(KeyError), guard():
        raise KeyError("a fault, not a refusal")
    # the help page (`python -m cortec`) lists every public callable with its real signature
    import cortec
    from cortec.__main__ import main as help_main
    assert help_main(["--help"]) == 0 and help_main(["-h"]) == 0
    capsys.readouterr()
    assert help_main([]) == 0
    page = capsys.readouterr().out
    assert "── CoRTeC" in page and "Help · What you can run" in page and "run order" in page
    for name in cortec.__all__:
        if callable(getattr(cortec, name, None)):
            assert name in page, name
    assert page.count("\n") < 60                                 # one screen, not a reference dump
    assert help_main(["bound_with_controls"]) == 0
    one = capsys.readouterr().out                               # arguments read from the code
    assert "Help · bound_with_controls" in one and "tolerance" in one and "auto" in one   # the tolerance is derived from the release by default
    assert help_main(["Generator.generate"]) == 0 and help_main(["nope"]) == 2
    with pytest.raises(SystemExit) as e:                        # `cortec run --help` is argparse's
        help_main(["run", "--help"])
    assert e.value.code == 0 and "--data" in capsys.readouterr().out
    # the words in the output are explained: the whole glossary, and one term by name
    assert help_main(["terms"]) == 0
    glossary = capsys.readouterr().out
    for term in ("tolerance", "ceiling", "floor", "discriminating", "thin", "1-way TV", "yield"):
        assert term in glossary, term
    assert help_main(["discriminating"]) == 0
    assert "the floor is not" in " ".join(capsys.readouterr().out.split())   # wrapped text


def test_save_writes_every_export_and_names_them(tmp_path):
    r = _record()
    files = r.save(str(tmp_path), "eval")
    for key in ("json", "markdown", "values_csv", "table_csv:fidelity and utility"):
        assert os.path.exists(files[key]), key
    assert "figure" in files
    if not files["figure"].startswith("not written"):
        assert files["figure"].endswith("eval.png") and os.path.getsize(files["figure"]) > 1000
    md = open(files["markdown"]).read()
    assert "| *real* |" in md and "**Verdict.** fine" in md
    assert RunRecord.from_json(files["json"]).files == files


def test_a_stage_without_a_figure_skips_it_without_failing(tmp_path):
    from cortec.generate import GenerationStats
    r = record_generation(GenerationStats(calls=2, parse_ok=2, rows=50, rows_requested=50), schema="toy")
    files = r.save(str(tmp_path), "gen")
    assert "figure" not in files and os.path.exists(files["json"])


# ── builders ──────────────────────────────────────────────────────────────────────────
def test_release_record_reads_the_audit_not_the_private_data():
    from cortec.release import Release
    rel = Release(schema_name="toy", epsilon_total=2.0, n_min=150,
                  cohorts=[{"cohort_id": 0, "cohort_name": "all", "cohort_size": 900,
                            "numerical": {}, "categorical": {}, "class_balance": {"YES": 0.2, "NO": 0.8},
                            "by_class": {}}],
                  conditional_levels=[{"level": 0, "columns": ["g"],
                                       "cells": {"g=a": {"rate": 0.1, "support": 400},
                                                 "g=b": {"rate": 0.3, "support": 500}}}],
                  audit={"epsilon_declared": 2.0, "epsilon_accounted": 2.0, "within_budget": True,
                         "max_rows_per_person": 1, "epsilon_per_person": 2.0, "n_queries": 7,
                         "groups": {"marginals": {"n_queries": 5, "n_partitions": 1, "group_total": 1.6}},
                         "warnings": ["seeded"]})
    r = record(rel, n_private_rows=1000)
    assert r.tool == "cortec" and r.stage == "Stage A"
    assert r.value("epsilon accounted") == 2.0 and r.value("cells at the finest level") == 2
    assert r.value("private rows read") == 1000 and r.warnings == ["seeded"]
    assert r.table("cohorts").rows[0][:2] == ["all", 900]
    assert r.table("conditional table, level 0").rows[1] == ["g=b", 0.3, 500]
    assert r.table("privacy accounting").rows[0] == ["marginals", 5, 1, 1.6]
    assert r.epsilon["per_row"] == 2.0 and r.verdict is None


def test_bound_record_carries_the_verdict_and_refuses_when_not_discriminating():
    from cortec.bound import BoundReport, BoundResult
    def res(cond, worst, within):
        return BoundResult(condition=cond, level=1, columns=["g", "h"], epsilon_cert=1.0, alpha=0.05,
                           n_cells=6, n_cells_covered=6, n_cells_uncovered=0, n_cells_thin=1,
                           worst_case_bound=worst, mean_bound=worst / 2, within_bound=within)
    claim = {"epsilon_release": 2.0, "epsilon_transmission_bound": 1.0, "epsilon_total_per_row": 3.0,
             "epsilon_per_person": 3.0, "max_rows_per_person": 1}
    good = BoundReport(synthetic=res("synthetic", 0.10, True), ceiling=res("ceiling", 0.08, True),
                       floor=res("floor", 0.30, False), tolerance=0.15, discriminating=True,
                       verdict="within bound", dp_claim=claim, accounting={}, meta={})
    r = record_bound(good, schema="toy")
    assert r.verdict_role == "ok" and "within bound" in r.verdict and r.schema == "toy"
    t = r.table("bound per condition")
    assert t.reference_rows == [1, 2] and t.rows[0][0] == "synthetic" and t.rows[0][6] == 0.10
    assert r.epsilon == {"release": 2.0, "bound": 1.0, "per_row": 3.0, "per_person": 3.0}
    bad = BoundReport(synthetic=res("synthetic", 0.10, True), ceiling=res("ceiling", 0.20, False),
                      floor=res("floor", 0.30, False), tolerance=0.15, discriminating=False,
                      verdict=None, dp_claim=dict(claim, epsilon_per_person_vacuous=True),
                      accounting={}, meta={})
    r = record_bound(bad)
    assert r.verdict_role == "warning" and r.verdict.startswith("no verdict")
    assert any("vacuous" in w for w in r.warnings)
    with pytest.raises(TypeError):
        record(object())


# ── evaluation ────────────────────────────────────────────────────────────────────────
def _toy(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    grp = rng.choice(["a", "b", "c"], n)
    p = 1 / (1 + np.exp(-(age - 50) / 8))
    y = np.where(rng.uniform(size=n) < p, "YES", "NO")
    return pd.DataFrame({"age": age, "grp": grp, "y": y})


def test_evaluate_scores_beside_a_real_sample_and_a_permuted_floor():
    pytest.importorskip("sklearn")
    from cortec import Schema, evaluate
    schema = Schema(name="toy", numerical={"age": (18, 80)}, categorical={"grp": ["a", "b", "c"]},
                    target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 80]})
    train, holdout = _toy(600, 1), _toy(300, 2)
    good = train.sample(200, random_state=5)
    missing = good.copy()
    missing["grp"] = "a"                       # a table that dropped two declared categories
    rec = evaluate(schema, {"copy": good, "collapsed": missing}, train=train, holdout=holdout)
    t = rec.table("fidelity and utility")
    assert [r[0] for r in t.rows][:2] == ["copy", "collapsed"] and t.reference_rows == [2, 3]
    assert t.columns == ["table", "rows", "1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM", "absent categories"]
    ceiling, floor = t.rows[2], t.rows[3]
    assert ceiling[3] > floor[3] + 0.15                  # the references separate on a real signal
    assert abs(t.rows[0][3] - ceiling[3]) < 0.1           # a copy of real data scores like real data
    assert t.rows[0][6] == 0 and t.rows[1][6] == 2        # absent categories are counted
    assert any("collapsed: absent categories grp=b, grp=c" in w for w in rec.warnings)
    assert rec.stage == "Evaluation" and rec.value("reference size n") == 200
    assert rec.value("reference draws") == 5 and rec.table("reference spread") is not None
    sp = rec.table("reference spread")
    assert [r[0] for r in sp.rows] == ["1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM"]
    for r in sp.rows:                                  # min <= mean <= max on both sides
        assert r[2] <= r[1] <= r[3] and r[5] <= r[4] <= r[6]
    assert sp.rows[1][2] > sp.rows[1][6]               # every real LR beats every permuted LR
    assert abs(ceiling[3] - sp.rows[1][1]) < 1e-9     # the table's reference row is the mean
    # one draw reproduces the single-sample layout, without the spread table
    one = evaluate(schema, {"copy": good}, train=train, holdout=holdout, reference_draws=1)
    assert one.table("reference spread") is None and one.value("reference draws") == 1
    assert one.table("fidelity and utility").rows[1][0] == "real sample, n=200 (ceiling)"
    with pytest.raises(ValueError):
        evaluate(schema, {"copy": good}, train=train, holdout=holdout, reference_draws=0)


# ── figures ───────────────────────────────────────────────────────────────────────────
def test_figures_use_the_paper_style_and_exist_for_the_stages_that_have_one(tmp_path):
    pytest.importorskip("matplotlib")
    from cortec import plots
    rc = plots.paper_rcparams()
    assert rc["font.family"] == "DejaVu Sans" and rc["axes.spines.top"] is False
    assert rc["axes.grid"] is True and rc["legend.frameon"] is False and rc["axes.titleweight"] == "bold"
    p = plots.figure_for(_record(), str(tmp_path / "e.png"))
    assert p and os.path.getsize(p) > 1000
    from cortec.generate import GenerationStats
    assert plots.figure_for(record_generation(GenerationStats()), str(tmp_path / "g.png")) is None


# ── the results panel ─────────────────────────────────────────────────────────────────
def _evaluation(syn, real, perm, spread=None):
    cols = ["table", "rows", "1-way TV", "TSTR-LR", "TSTR-RF", "TSTR-GBM", "absent categories"]
    t = ResultTable("fidelity and utility", cols,
                    [["synthetic", 300] + syn, ["real sample, n=300 (ceiling)", 300] + real,
                     ["permuted target, n=300 (floor)", 300] + perm], reference_rows=[1, 2])
    tables = [t]
    if spread:
        tables.append(ResultTable("reference spread", ["measure", "real sample mean", "real min", "real max",
                                                       "permuted mean", "permuted min", "permuted max"], spread))
    return RunRecord(tool="cortec", stage="Evaluation", title="x", schema="toy", tables=tables)


SPREAD = [["1-way TV", 0.042, 0.032, 0.046, 0.042, 0.032, 0.046],
          ["TSTR-LR", 0.758, 0.740, 0.764, 0.498, 0.456, 0.549],
          ["TSTR-RF", 0.718, 0.692, 0.739, 0.538, 0.497, 0.568],
          ["TSTR-GBM", 0.713, 0.656, 0.751, 0.495, 0.444, 0.517]]


def test_results_panel_reads_each_goal_against_the_real_sample_range():
    ev = _evaluation([0.044, 0.711, 0.700, 0.760, 0], [0.042, 0.758, 0.718, 0.713, 0],
                     [0.042, 0.498, 0.538, 0.495, 0], SPREAD)
    r = record_results(evaluation=ev, schema="toy")
    g1, g2 = r.table("marginal fidelity result"), r.table("downstream utility result")
    assert g1.rows[0][1] == 0.044 and g1.note.startswith("reading: within the real-sample range (0.032 to 0.046)")
    assert [row[0] for row in g2.rows] == ["LR", "RF", "GBM"] and g2.subtitle.startswith("Does a model")
    assert g2.rows[0][5] == "82%"                      # (0.711 - 0.498) / (0.758 - 0.498)
    assert "LR: below the real-sample range by 0.029" in g2.note    # the lowest real sample is 0.740
    assert "GBM: above every real sample of this size" in g2.note
    assert r.verdict_role == "warning" and "downstream utility (LR below the range by 0.029)" in r.verdict
    v = dict(r.values)
    assert v["marginal fidelity"].startswith("within the real-sample range")
    assert v["downstream utility"].startswith("LR below the real-sample range (by 0.029)")
    assert "the other students within it" in v["downstream utility"]
    # a panel inside the range on every measure reads ok, and says so once
    ev = _evaluation([0.040, 0.750, 0.700, 0.700, 0], [0.042, 0.758, 0.718, 0.713, 0],
                     [0.042, 0.498, 0.538, 0.495, 0], SPREAD)
    r = record_results(evaluation=ev)
    assert r.verdict_role == "ok" and "within or beyond the range of real samples" in r.verdict
    assert dict(r.values)["downstream utility"] == "every student within or beyond the real-sample range"
    # the table subtitles survive the JSON round trip
    back = RunRecord.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.table("marginal fidelity result").subtitle == g1.subtitle
    assert "*Does the synthetic table" in back.to_markdown()


def test_results_panel_with_one_reference_draw_reports_gaps_not_ranges():
    ev = _evaluation([0.044, 0.711, 0.700, 0.760, 0], [0.042, 0.758, 0.718, 0.713, 0],
                     [0.042, 0.498, 0.538, 0.495, 0])
    r = record_results(evaluation=ev)
    assert r.verdict_role == "ok" and "is +0.002 on marginal fidelity" in r.verdict and "at most 0.047 AUC on downstream utility (LR)" in r.verdict
    assert dict(r.values)["downstream utility"] == "LR gap -0.047; RF gap -0.018; GBM gap +0.047"


def test_results_panel_without_an_evaluation_says_so_and_still_carries_bound_privacy_and_cost():
    b = RunRecord(tool="cortec", stage="Stage C", title="x", schema="toy",
                  values=[("tolerance", "0.0879")], verdict="within bound: over all 5 released cells ...",
                  verdict_role="ok", epsilon={"bound": 1.0, "per_row": 3.0})
    a = RunRecord(tool="cortec", stage="Stage A", title="x", schema="toy", epsilon={"accounted": 2.0})
    g = RunRecord(tool="cortec", stage="Stage B", title="x", schema="toy",
                  values=[("calls", 24), ("spend (USD)", 0.74), ("budget capped", True)])
    r = record_results(evaluation=None, bound=b, generation=g, release=a, seconds=75)
    v = dict(r.values)
    assert v["marginal fidelity"] == "not scored in this run" and r.tables == [] and r.verdict is None
    assert any("did not run" in n for n in r.notes)
    assert v["Stage C, transmission bound"] == "within bound at tolerance 0.0879"
    assert v["privacy spent"].startswith("epsilon 2.000 for the release + 1.000 for the bound = 3.000 per row")
    assert v["cost"] == "24 model calls, $0.74, budget cap reached, 1 min 15 s in all"
    b.verdict, b.verdict_role = "no verdict: the controls did not discriminate: ...", "warning"
    assert dict(record_results(bound=b).values)["Stage C, transmission bound"].startswith("no verdict")


def test_banner_files_block_and_wrapping_keep_paths_whole():
    # the method is written CoRTeC wherever the tool names itself; the identifier stays `cortec`
    assert report.display_name("cortec") == "CoRTeC" and report.display_name("cortec-hybrid") == "CoRTeC-hybrid"
    assert "## CoRTeC " in RunRecord(tool="cortec", stage="Stage A", title="x", schema="t").to_markdown()
    path = "/a/very/long/path/" + "x" * 90
    b = report.banner("toy", [("output", path), ("stages", "A release, B generate")])
    assert b.startswith("══ CoRTeC") and path in b and "  stages  A release, B generate" in b
    f = report.files_block("everything", [("folder", "/p"), ("RESULTS.md", "the panel")])
    assert f.startswith("── CoRTeC") and "  folder      /p" in f and "  RESULTS.md  the panel" in f
    long = "word " * 40
    assert all(len(line) <= report.WIDTH for line in report.note(long.strip()).split("\n"))
    assert fmt_seconds(12) == "12 s" and fmt_seconds(75) == "1 min 15 s" and fmt_seconds(3723) == "1 h 02 min"


def test_bound_verdicts_say_what_was_bounded(toy_bound_report=None):
    """The verdict sentence names the tolerance to four places, the cells and the confidence, and a
    no-verdict names the control that failed."""
    class _R:
        n_cells, n_cells_covered, n_cells_uncovered, n_cells_thin = 3, 3, 0, 0
        mean_bound, alpha, level, columns = 0.03, 0.05, 1, ["age"]
        def __init__(self, worst, within):
            self.worst_case_bound, self.within_bound = worst, within
    class _Rep:
        dp_claim = {"epsilon_transmission_bound": 1.0, "epsilon_release": 2.0, "epsilon_total_per_row": 3.0,
                    "epsilon_per_person": 3.0}
        meta = {"schema": "toy"}
        tolerance, discriminating, tolerance_rule = 0.0879, True, {"rule": "derived from the release"}
        synthetic, ceiling, floor = _R(0.04, True), _R(0.06, True), _R(0.15, False)
        verdict, verdict_reason = "within bound", None
    rec = record_bound(_Rep())
    assert rec.verdict.startswith("within bound: over all 3 released cells") and "0.0879" in rec.verdict
    assert "95% simultaneous confidence" in rec.verdict and dict(rec.values)["tolerance"] == "0.0879"
    bad = _Rep(); bad.verdict, bad.verdict_reason = None, "the controls did not discriminate: the permuted-target floor's worst bound 0.0500 lies within the tolerance 0.0879; no verdict is issued"
    rec = record_bound(bad)
    assert rec.verdict_role == "warning" and rec.verdict.startswith("no verdict: the controls did not discriminate")
    assert rec.notes[0].startswith("A no verdict means the test could not decide")
