# -*- coding: utf-8 -*-
"""Unit-Tests für notebook_hub.py — mit pytest UND unittest ausführbar."""
from __future__ import annotations

import json
import sys
import unittest
import warnings
import zipfile
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notebook_hub as nh  # noqa: E402


class WorkspaceTestCase(unittest.TestCase):
    """Jeder Test läuft in einem frischen, nach dem Test gelöschten Workspace."""

    def setUp(self) -> None:
        self._workspace_ctx = nh.temporary_workspace()
        self.workspace = self._workspace_ctx.__enter__()
        self.addCleanup(self._workspace_ctx.__exit__, None, None, None)

    def write(self, rel: str, content: str) -> Path:
        path = nh.CFG.target_dir / rel
        nh.write_file_atomic(path, content)
        return path

    def read(self, rel: str) -> str:
        return nh.read_text_safe(nh.CFG.target_dir / rel)


class TestPathSafety(WorkspaceTestCase):
    def test_path_safety(self):
        for value in ("../../etc/passwd", "..\\..\\windows\\system32", "~/secret.py",
                      "a/../../b.py", "", None):
            with self.subTest(value=value):
                self.assertIsNone(nh.normalize_rel_path(value))

    def test_normal_paths_are_normalized(self):
        self.assertEqual(nh.normalize_rel_path("Test1/pkg/mod.py"), "pkg/mod.py")
        self.assertEqual(nh.normalize_rel_path("./pkg\\mod.py"), "pkg/mod.py")
        self.assertEqual(nh.normalize_rel_path(" `src/app.py` "), "src/app.py")
        self.assertIsNone(nh.normalize_rel_path("C:/project/app.py"))

    def test_workspace_resolution(self):
        self.assertIsNone(nh.resolve_in_workspace("../outside.py"))
        inside = nh.resolve_in_workspace("pkg/ok.py")
        self.assertIsNotNone(inside)
        self.assertIn(str(nh.CFG.target_dir.resolve()), str(inside.resolve()))
        absolute = nh.resolve_in_workspace(str(nh.CFG.target_dir / "pkg" / "absolute.py"))
        self.assertIsNotNone(absolute)
        self.assertEqual(absolute.name, "absolute.py")
        self.assertIsNone(nh.resolve_in_workspace("/etc/passwd"))


class TestParsing(WorkspaceTestCase):
    def test_code_block_extraction(self):
        self.assertEqual(nh.extract_code_block("text\n```python\nx = 1\n```\nend"), "x = 1")
        self.assertEqual(nh.extract_code_block("```json\n{}\n```\n```python\ny=2\n```", "python"), "y=2")
        self.assertEqual(nh.extract_code_block("```python\nprint('open')"), "print('open')")
        self.assertEqual(nh.extract_code_block("plain code"), "plain code")
        self.assertEqual(len(nh.extract_all_code_blocks("```python\nx=1\n```\n```sh\necho ok\n```")), 2)
        self.assertEqual(nh.strip_thinking("<think>hidden</think>visible"), "visible")

    def test_parse_file_blocks_formats(self):
        cases = (
            ("### FILE: pkg/math.py\n```python\ndef f(): return 1\n```", "pkg/math.py", "python"),
            ("FILE: README.md\n```markdown\n# Hello\n```", "README.md", "markdown"),
            ("**Datei:** `config.json`\n```json\n{\"x\": 1}\n```", "config.json", "json"),
        )
        for answer, expected_path, expected_language in cases:
            with self.subTest(path=expected_path):
                parsed = nh.parse_file_blocks(answer)
                self.assertEqual(len(parsed), 1)
                self.assertEqual(parsed[0].path, expected_path)
                self.assertEqual(parsed[0].language, expected_language)

    def test_parser_preserves_nested_markdown_fences(self):
        markdown = "# Example\n\n```python\nprint('hello')\n```\n"
        answer = nh.render_file_blocks({"guide.md": markdown})
        parsed = nh.parse_file_blocks(answer)
        self.assertEqual(parsed[0].content, markdown)
        self.assertEqual(parsed[0].path, "guide.md")

    def test_parser_rejects_traversal_and_merges_duplicates(self):
        unsafe = nh.parse_file_blocks("### FILE: ../../evil.py\n```python\nx=1\n```")
        self.assertTrue(unsafe)
        self.assertEqual(unsafe[0].path, "")
        dup = nh.parse_file_blocks(
            "### FILE: value.py\n```python\nVALUE=1\n```\n"
            "### FILE: value.py\n```python\nVALUE=2\n```"
        )
        self.assertEqual(len(dup), 1)
        self.assertIn("VALUE=2", dup[0].content)


class TestSyntaxAndWrites(WorkspaceTestCase):
    def test_mechanical_repair(self):
        fixed, notes = nh.mechanical_repair("```python\nvalue = [1, 2\n```\n")
        self.assertTrue(nh.check_syntax(fixed).ok, fixed)
        self.assertTrue(notes)

    def test_syntax_and_lint(self):
        self.assertTrue(nh.check_syntax("def add(a, b):\n    return a + b\n").ok)
        bad = nh.check_syntax("def add(:\n    return 1\n")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.lineno, 1)
        report = nh.lint_summary("import os\nimport sys\n\ndef f():\n    return os.getcwd()\n")
        self.assertEqual(report["functions"], 1)
        self.assertIn("sys", report["unused_imports"])

    def test_writer_backs_up_and_rejects_invalid_syntax(self):
        self.write("mod.py", "VALUE = 1\n")
        good = nh.ParsedFile("mod.py", "mod.py", "VALUE = 2\n", "python")
        results, _ = nh.write_generated_files([good], backend=nh.MockBackend())
        self.assertEqual(results[0].action, "überschrieben")
        self.assertTrue(results[0].backup)
        self.assertEqual(self.read("mod.py"), "VALUE = 2\n")
        bad = nh.ParsedFile("mod.py", "mod.py", "def broken(:\n", "python")
        results, _ = nh.write_generated_files([bad], auto_fix=False)
        self.assertIn("abgelehnt", results[0].action)
        self.assertEqual(self.read("mod.py"), "VALUE = 2\n")

    def test_ast_self_healing_offline(self):
        self.write("broken.py", "```python\ndef add(a, b):\n    return a + b\n```\n")
        ok, message = nh.auto_fix_code_loop("broken.py", backend=nh.MockBackend())
        self.assertTrue(ok, message)
        self.assertTrue(nh.check_syntax(self.read("broken.py")).ok)
        self.assertIn("def add", self.read("broken.py"))


class TestWorkspaceFiles(WorkspaceTestCase):
    def test_workspace_manager(self):
        self.assertIn("Angelegt", nh.create_new_file("mod.py", "Modul (Python)"))
        self.assertIn("bereits", nh.create_new_file("mod.py").lower())
        self.assertIn("Gespeichert", nh.save_file_content("mod.py", "def answer():\n    return 42\n"))
        self.assertIn("answer", nh.get_file_content("mod.py"))
        self.assertIn("1 Treffer", nh.search_workspace("return 42"))
        self.assertIn("DATEI-DETAILS", nh.file_details("mod.py"))
        self.assertIn("Umbenannt", nh.rename_workspace_file("mod.py", "src/mod.py"))
        self.assertIn("Gelöscht", nh.delete_workspace_file("src/mod.py", confirm=True))
        self.assertIn("bestätigungs-haken", nh.delete_workspace_file("absent.py").lower())

    def test_zip_import_safety(self):
        archive_path = nh.CFG.workspace_dir / "source.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("wrapper/mod.py", "VALUE = 7\n")
            archive.writestr("wrapper/../../escape.py", "PWNED = True\n")
        message = nh.import_into_workspace([str(archive_path)])
        self.assertIn("ZIP entpackt", message)
        self.assertTrue((nh.CFG.target_dir / "mod.py").exists())
        self.assertFalse((nh.CFG.workspace_dir / "escape.py").exists())
        exported = nh.export_workspace_zip()
        self.assertTrue(exported and zipfile.is_zipfile(exported))
        with zipfile.ZipFile(exported) as archive:
            self.assertIn("mod.py", archive.namelist())

    def test_tree_hides_internal_files(self):
        self.write("visible.py", "pass\n")
        self.write(".secret", "secret\n")
        self.write("__pycache__/cache.py", "pass\n")
        tree, files, count = nh.build_file_tree()
        self.assertEqual(count, 1)
        self.assertEqual(files, ["visible.py"])
        self.assertNotIn(".secret", tree)

    def test_search_handles_regex_errors(self):
        self.write("a.py", "alpha\nbeta\n")
        self.assertIn("1 Treffer", nh.search_workspace("alp"))
        self.assertIn("Ungültiger regulärer Ausdruck", nh.search_workspace("[", use_regex=True))

    def test_bounded_text_read(self):
        path = self.write("large.txt", "x" * 20000)
        content = nh.read_text_safe(path, max_chars=500)
        self.assertLess(len(content), 600)
        self.assertIn("gekürzt", content)


class TestExecutionAndTests(WorkspaceTestCase):
    def test_execution_engine(self):
        self.write("ok.py", "print('hello')\n")
        output = nh.execute_python_file("ok.py")
        self.assertIn("hello", output)
        self.assertIn("Return Code: 0", output)
        self.write("exit.py", "raise SystemExit(4)\n")
        self.assertIn("Return Code: 4", nh.execute_python_file("exit.py"))
        self.write("stderr.py", "import sys\nprint('err', file=sys.stderr)\n")
        self.assertIn("=== STDERR ===", nh.execute_python_file("stderr.py"))
        self.assertIn("err", nh.execute_python_file("stderr.py"))
        self.write("notes.txt", "not executable\n")
        self.assertIn("Kein Interpreter", nh.execute_python_file("notes.txt"))
        self.assertIn("Datei nicht gefunden", nh.execute_python_file("missing.py"))

    def test_execution_timeout_and_live_stream(self):
        self.write("loop.py", "import time\nwhile True: time.sleep(0.02)\n")
        output = nh.execute_python_file("loop.py", timeout=1)
        self.assertIn("ABGEBROCHEN", output)
        self.write("hello.py", "print('streamed')\n")
        chunks = list(nh.stream_execution("hello.py"))
        self.assertIn("streamed", chunks[-1])
        self.assertIn("Return Code: 0", chunks[-1])

    def test_demo_project_and_tests_pass(self):
        progress = list(nh.iter_synthesis(
            user_objective="Generate the offline demo project and verify its tests",
            auto_gen_tests=True, run_tests=True, heal=False, use_mock=True, runner="auto",
        ))
        self.assertTrue(progress)
        final = progress[-1]
        self.assertTrue(final.report and final.report.ok,
                        final.test_summary + "\n" + (final.report.raw_output[-1500:] if final.report else ""))
        self.assertGreaterEqual(final.report.total, 10)

    def test_zero_tests_is_not_success(self):
        report = nh.TestReport(runner="unittest", returncode=0, total=0, collected=True)
        self.assertFalse(report.ok)

    def test_unittest_discovery_command_order(self):
        command = nh.build_test_command("unittest")
        self.assertTrue(command[1].endswith("unittest_runner.py"))
        self.write("test_one.py", "import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n")
        single = nh.build_test_command("unittest", target="test_one.py")
        self.assertTrue(single[-1].endswith("test_one.py"))
        with self.assertRaises(ValueError):
            nh.build_test_command("pytest", target="../README.md")
        report = nh.TestReport(runner="unittest", total=2, passed=1, failed=1,
                               collected=True, returncode=1)
        self.assertFalse(report.ok)

    def test_unittest_fallback_discovers_nested_tests_without_init(self):
        self.write("src/value.py", "def answer(): return 42\n")
        self.write("src/test_value.py", "import unittest\nimport value\n"
                   "class T(unittest.TestCase):\n    def test_answer(self): self.assertEqual(value.answer(), 42)\n")
        report = nh.run_tests_structured(runner="unittest")
        self.assertTrue(report.ok, report.summary() + "\n" + report.raw_output)
        self.assertEqual(report.total, 1)

    def test_test_parsers(self):
        output = """test_one (test_mod.TestMod.test_one) ... FAIL

======================================================================
FAIL: test_one (test_mod.TestMod.test_one)
----------------------------------------------------------------------
Traceback (most recent call last):
  File \"test_mod.py\", line 4, in test_one
AssertionError: no

----------------------------------------------------------------------
Ran 1 test in 0.01s

FAILED (failures=1)
"""
        report = nh.parse_unittest_text(output)
        self.assertEqual((report.total, report.failed, report.passed), (1, 1, 0))
        skipped = nh.parse_unittest_text("Ran 2 tests in 0.01s\n\nOK (skipped=1)\n")
        self.assertEqual((skipped.total, skipped.skipped, skipped.passed), (2, 1, 1))
        pytest_report = nh.parse_pytest_text("F. [100%]\nFAILED test_mod.py::test_a - assert 1 == 2\n1 failed, 1 passed in 0.03s\n")
        self.assertEqual((pytest_report.total, pytest_report.failed, pytest_report.passed), (2, 1, 1))

    def test_test_output_generator_yields_live_updates(self):
        files = nh.demo_project()
        nh.write_generated_files(nh.parse_file_blocks(nh.render_file_blocks(files)),
                                 backend=nh.MockBackend())
        updates = list(nh.stream_workspace_tests(runner="unittest"))
        self.assertGreaterEqual(len(updates), 2)
        self.assertIn("Fertig", updates[-1][1])
        self.assertTrue(updates[-1][2].ok, updates[-1][2].summary())


class TestGenerationAndHealing(WorkspaceTestCase):
    def test_offline_auto_test_generation(self):
        self.write("sample.py", "def square(value):\n    return value * value\n")
        ok, summary = nh.generate_tests_for("sample.py", backend=nh.MockBackend())
        self.assertTrue(ok, summary)
        self.assertTrue((nh.CFG.target_dir / "test_sample.py").exists())
        content = self.read("test_sample.py")
        self.assertIn("import sample", content)
        self.assertIn("test_square_callable", content)

    def test_pipeline_generates_missing_companion_tests_for_each_source(self):
        class NoTestsBackend(nh.MockBackend):
            def _answer(self, prompt, system=""):
                if "[TASK:SYNTHESIS]" in prompt:
                    files = {name: content for name, content in nh.demo_project().items()
                             if not nh._is_test_path(name)}
                    return nh.render_file_blocks(files, intro="No tests included by the model.")
                return super()._answer(prompt, system)

        backend = NoTestsBackend()
        with patch.object(nh, "get_backend", return_value=backend):
            progress = list(nh.iter_synthesis(
                user_objective="Generate a small project",
                auto_gen_tests=True, run_tests=True, heal=False, use_mock=True,
            ))
        final = progress[-1]
        test_files = set(nh.scan_workspace().test_files)
        self.assertTrue(final.report and final.report.ok, final.test_summary)
        self.assertTrue({"test_app.py", "test_mathlib.py", "test_textlib.py"}.issubset(test_files))
        generated = {row[0] for row in final.rows}
        self.assertIn("test_app.py", generated)

    def test_self_healing_fixes_logic_bugs(self):
        files = nh.demo_project(inject_bugs=True)
        results, _ = nh.write_generated_files(nh.parse_file_blocks(nh.render_file_blocks(files)),
                                              backend=nh.MockBackend(), auto_fix=False)
        self.assertTrue(any(result.action == "erstellt" for result in results))
        before = nh.run_tests_structured(runner="unittest")
        self.assertGreater(before.bad_count, 0, before.summary())
        healing = nh.run_self_healing(backend=nh.MockBackend(), max_rounds=2,
                                      runner="unittest", report=before)
        self.assertTrue(healing.success, healing.summary() + "\n" + healing.log)
        self.assertTrue(healing.changed_files)
        self.assertTrue(all(not nh._is_test_path(path) for path in healing.changed_files))
        self.assertTrue(nh.run_tests_structured(runner="unittest").ok)

    def test_self_healing_rollback_and_anticheat(self):
        self.write("test_guard.py", "import unittest\nclass T(unittest.TestCase):\n def test_x(self): self.assertEqual(1, 2)\n")
        self.write("guard.py", "VALUE = 1\n")
        parsed = [nh.ParsedFile("test_guard.py", "test_guard.py", "VALUE = 2\n", "python")]
        changed, _, notes = nh._apply_repair_files(parsed, allow_test_edits=False, round_tag="guard")
        self.assertEqual(changed, [])
        self.assertTrue(any("BLOCKIERT" in item for item in notes))
        self.assertIn("assertEqual", self.read("test_guard.py"))

        # Neu angelegte Reparaturdateien müssen beim Rollback ebenfalls verschwinden.
        new_file = nh.CFG.target_dir / "new_module.py"
        changed, backups, _notes = nh._apply_repair_files(
            [nh.ParsedFile("new_module.py", "new_module.py", "VALUE = 1\n", "python")],
            allow_test_edits=False, round_tag="new-file")
        self.assertEqual(changed, ["new_module.py"])
        self.assertTrue(new_file.exists())
        self.assertTrue(nh._rollback(backups, changed_paths=changed))
        self.assertFalse(new_file.exists())


class TestUICallbacks(WorkspaceTestCase):
    def test_verdict_card_marks_failure_and_escapes_values(self):
        failed = nh.TestReport(runner="<script>alert(1)</script>", returncode=1,
                               total=4, passed=2, failed=2, collected=True)
        markup = nh.render_test_verdict_html(failed, "Fertig")
        self.assertIn("nh-test-verdict--failed", markup)
        self.assertIn("Testfehler erkannt", markup)
        self.assertIn("Fehlgeschlagen", markup)
        self.assertIn("Self-Healing-Loop", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertNotIn("<script>", markup)

        passed = nh.TestReport(runner="pytest", returncode=0, total=4,
                               passed=4, collected=True, duration=0.25)
        green_markup = nh.render_test_verdict_html(passed, "Fertig")
        self.assertIn("nh-test-verdict--passed", green_markup)
        self.assertIn("Alle Tests bestanden", green_markup)

    def test_test_callback_highlights_a_real_failure(self):
        self.write(
            "test_failure.py",
            "import unittest\nclass FailureCase(unittest.TestCase):\n"
            "    def test_expected_red_state(self): self.assertEqual(1, 2)\n",
        )
        updates = list(nh.ui_run_tests(runner="unittest", timeout=30))
        self.assertGreaterEqual(len(updates), 2)
        final = updates[-1]
        self.assertIn("nh-test-verdict--failed", final[1])
        self.assertIn("Testfehler erkannt", final[1])
        self.assertIn("test_expected_red_state", final[3])

    def test_workspace_callback_smoke(self):
        tree, choices, stats = nh.ui_ws_refresh()
        self.assertIsInstance(tree, str)
        self.assertIsNotNone(choices)
        self.assertIn("Dateien", stats)

        status, tree, choices, stats = nh.ui_ws_create("callback.py", "Modul (Python)")
        self.assertIn("Angelegt", status)
        self.assertIn("callback.py", tree)
        self.assertIsNotNone(choices)
        self.assertIn("Dateien", stats)

        content, details, download, note = nh.ui_ws_select("callback.py")
        self.assertIn("def main", content)
        self.assertIn("DATEI-DETAILS", details)
        self.assertTrue(str(download).endswith("callback.py"))
        self.assertIn("callback.py", note)

        status, tree, choices, _stats = nh.ui_ws_save("callback.py", "print('saved by callback')\\n")
        self.assertIn("Gespeichert", status)
        self.assertIn("callback.py", tree)
        self.assertIsNotNone(choices)
        self.assertIn("1 Treffer", nh.ui_ws_search("saved by callback", False, False, ".py"))

        status, tree, choices, _stats = nh.ui_ws_rename("callback.py", "renamed.py")
        self.assertIn("Umbenannt", status)
        self.assertIn("renamed.py", tree)
        self.assertIsNotNone(choices)
        status, tree, choices, _stats = nh.ui_ws_delete("renamed.py", True)
        self.assertIn("Gelöscht", status)
        self.assertNotIn("renamed.py", tree)
        self.assertIsNotNone(choices)

    def test_logs_callback_smoke(self):
        nh.LOG.info("UI callback smoke entry", "test")
        log_text, audit = nh.ui_logs_refresh("INFO", 20, "UI callback smoke")
        self.assertIn("UI callback smoke entry", log_text)
        self.assertIsInstance(audit, dict)
        self.assertIn("entries", audit)
        self.assertGreaterEqual(audit["count"], 1)


class TestInternalSelftest(WorkspaceTestCase):
    def test_internal_selftest_passes(self):
        passed, failed, report = nh.run_internal_selftest(verbose=False)
        self.assertGreater(passed, 0, report)
        self.assertEqual(failed, 0, report)
        self.assertIn("ALLE CHECKS GRÜN", report)


class TestChatConfigAndLogging(WorkspaceTestCase):
    def test_offline_dashboard_is_not_a_false_failure(self):
        original = nh.CFG.offline_demo
        nh.CFG.offline_demo = True
        self.addCleanup(setattr, nh.CFG, "offline_demo", original)
        with patch.object(nh.OLLAMA, "ping", return_value={"ok": False, "detail": "Ollama unavailable"}), \
             patch.object(nh.OLLAMA, "list_models", return_value=[]), \
             patch.object(nh.OLLAMA, "running_models", return_value=[]), \
             patch.object(nh.OLLAMA, "model_info", return_value=[]):
            status = nh.ollama_health_text()
            models = nh.model_table_rows(force=False)
        self.assertIn("Offline-Demo aktiv", status)
        self.assertIn("Laufmodus", status)
        self.assertNotIn("🔴", status)
        self.assertEqual(models[0][0], "offline-demo")

    def test_history_normalization(self):
        history = nh.normalize_history([("hi", "hello")])
        self.assertEqual([entry["role"] for entry in history], ["user", "assistant"])
        self.assertEqual(nh.normalize_history([{"role": "user", "content": [{"text": "hi"}]}]),
                         [{"role": "user", "content": "hi"}])
        self.assertIn("hello", nh.format_chat_transcript(history))
        self.assertEqual(nh.trim_history([{"role": "user", "content": str(i)} for i in range(100)], 3).__len__(), 6)

    def test_offline_chat_streams(self):
        chunks = list(nh.stream_ollama("Explain a test", model="offline-demo", backend=nh.MockBackend()))
        self.assertTrue(chunks)
        self.assertIn("OFFLINE-DEMO", chunks[-1])

    def test_logs_filters_audit_clear(self):
        nh.LOG.info("a known info entry", "test")
        nh.LOG.error("a known error entry", "test")
        self.assertIn("known error", nh.LOG.read(level="ERROR"))
        self.assertNotIn("known info", nh.LOG.read(level="ERROR"))
        self.assertTrue(nh.LOG.read_audit())
        self.assertIn("geleert", nh.LOG.clear())
        self.assertIn("Keine Fehler", nh.LOG.read())

    def test_dashboard_payload_is_serializable(self):
        payload = nh.dashboard_json_payload()
        self.assertIsInstance(json.dumps(payload, ensure_ascii=False), str)
        self.assertIn("workspace", payload)

    @unittest.skipUnless(nh.GRADIO_AVAILABLE, "Gradio ist nicht installiert")
    def test_ui_builds(self):
        # Gradio-Component-Konstruktoren besitzen teilweise eigene Event-Loops,
        # die nicht geschlossene ResourceWarnings produzieren.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            demo = nh.build_ui()
        self.assertGreater(len(demo.blocks), 50)
        self.assertGreater(len(demo.fns), 10)
        markdown_values = [component.get("props", {}).get("value", "")
                           for component in demo.config.get("components", [])
                           if component.get("type") == "markdown"]
        header = next(value for value in markdown_values if "notebook_hub v" in str(value))
        help_text = next(value for value in markdown_values if "Umgebungsvariablen" in str(value))
        self.assertIn(f"Workspace `{nh.CFG.target_name}`", header)
        self.assertIn("Ollama Multi-Interface Notebook & Dev-Agent Studio", header)
        self.assertNotIn("{target}", header)
        self.assertNotIn("{target}", help_text)
        self.assertNotIn("&amp;", header)
        tabs = next(component for component in demo.config.get("components", [])
                    if component.get("type") == "tabs")
        if nh.GRADIO_MAJOR >= 6:
            self.assertEqual(tabs.get("props", {}).get("overflow_behavior"), "wrap")
        else:
            # Älteres Gradio 5 kennt overflow_behavior noch nicht; mk() filtert es heraus.
            self.assertNotIn("overflow_behavior", tabs.get("props", {}))
        toolbar_rows = [component for component in demo.config.get("components", [])
                        if "hub-toolbar-row" in component.get("props", {}).get("elem_classes", [])]
        self.assertEqual(len(toolbar_rows), 3)
        compact_upload = next(component for component in demo.config.get("components", [])
                              if "hub-upload-compact" in component.get("props", {}).get("elem_classes", []))
        self.assertEqual(compact_upload.get("props", {}).get("height"), 180)
        self.assertEqual(compact_upload.get("props", {}).get("label"),
                         "Spezifikationen und Projektdateien hochladen")
        input_row = next(component for component in demo.config.get("components", [])
                         if "agent-input-row" in component.get("props", {}).get("elem_classes", []))
        output_row = next(component for component in demo.config.get("components", [])
                          if "agent-output-row" in component.get("props", {}).get("elem_classes", []))
        self.assertEqual(input_row.get("props", {}).get("equal_height"), False)
        self.assertEqual(output_row.get("props", {}).get("equal_height"), False)
        self.assertIn("@media (max-width: 980px)", nh.APP_CSS)
        self.assertIn(".agent-input-row, .agent-output-row", nh.APP_CSS)
        load_dependency = next(dependency for dependency in demo.config.get("dependencies", [])
                               if any(len(target) > 1 and target[1] == "load"
                                      for target in dependency.get("targets", [])))
        self.assertFalse(load_dependency.get("queue", True))
        timers = [component for component in demo.config.get("components", [])
                  if component.get("type") == "timer"]
        self.assertTrue(all(not component.get("props", {}).get("active", True)
                            for component in timers))
        operations_table = next(component for component in demo.config.get("components", [])
                                if component.get("type") == "dataframe"
                                and component.get("props", {}).get("label") == "Datei-Operationen")
        self.assertFalse(operations_table.get("props", {}).get("wrap", True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
