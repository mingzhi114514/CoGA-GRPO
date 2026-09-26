from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseStructureTests(unittest.TestCase):
    def test_controlled_launcher_has_safe_default_method(self) -> None:
        script = (ROOT / "run_scripts" / "run_e0_e6.sh").read_text(encoding="utf-8")
        self.assertIn("REWARD_LONG_INCORRECT=false", script)
        self.assertIn("BETA_DIRECTION=lower_for_hard", script)
        self.assertIn('REWARD_PROTOCOL="rule_based_format_accuracy"', script)
        self.assertNotIn("LLM judge", script)

    def test_sft_stage_is_retained(self) -> None:
        launcher = (ROOT / "run_scripts" / "run_sft.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "data_config" / "stage1.yaml").read_text(encoding="utf-8")
        self.assertIn("src/open_r1/sft.py", launcher)
        self.assertIn("--freeze_vision_modules true", launcher)
        self.assertIn("--use_audio_in_video true", launcher)
        self.assertIn("Video-R1_rewrite.json", manifest)

    def test_full_method_switches_are_explicit(self) -> None:
        script = (ROOT / "run_scripts" / "run_e0_e6.sh").read_text(encoding="utf-8")
        expected = "E3) USE_DYNAMIC_BETA=true; ENABLE_LENGTH_REWARD=true ;;"
        self.assertIn(expected, script)

    def test_evaluation_wrapper_requires_complete_results(self) -> None:
        script = (ROOT / "run_scripts" / "eval_one.sh").read_text(encoding="utf-8")
        self.assertIn("eval/evaluate_coga.py", script)
        self.assertIn("--strict-completeness", script)
        self.assertIn("--no-skip-oom", script)
        self.assertIn("--video-max-frames 4", script)
        self.assertIn("--max-pixels 75264", script)
        self.assertIn("--min-pixels 50176", script)
        self.assertIn("--no-skip-missing-video-samples", script)
        self.assertIn("--no-do-sample", script)

    def test_readme_reproduces_main_method_without_identity_links(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("run_e0_e6.sh E3 full 42", readme)
        self.assertIn("e3_full_seed42", readme)
        self.assertNotIn("run_e0_e6.sh E4 full 42", readme)
        self.assertNotIn("CoGA-GRPO/actions/workflows", readme)

    def test_private_training_json_is_not_tracked_by_default(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data_config/*.json", ignore)
        self.assertIn("*.safetensors", ignore)

    def test_public_branding_is_coga_grpo(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertTrue(readme.startswith("# CoGA-GRPO"))
        self.assertIn('name = "coga-grpo"', metadata)


if __name__ == "__main__":
    unittest.main()
