#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "workflows/xpu_alignment.yml"
BOT_WORKFLOW = Path(__file__).parents[1] / "workflows/bot.yml"


class AlignmentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.bot_text = BOT_WORKFLOW.read_text(encoding="utf-8")

    def test_manual_dispatch_only_exposes_scan_date(self) -> None:
        dispatch = self.text.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        self.assertIn("scan_date:", dispatch)
        self.assertNotIn("runner:", dispatch)
        self.assertNotIn("auto_file:", dispatch)
        self.assertNotIn("notify:", dispatch)

    def test_xpu_jobs_use_the_dedicated_runner(self) -> None:
        self.assertEqual(self.text.count("runs-on: xpu-agent"), 2)
        self.assertNotIn("runs-on: ${{ inputs.runner", self.text)

    def test_collector_is_bounded_and_precedes_every_agent(self) -> None:
        collector = self.text.split("  collect:", 1)[1].split("  scan-prepare:", 1)[0]
        self.assertIn("runs-on: ubuntu-latest", collector)
        self.assertIn("timeout-minutes: 30", collector)
        self.assertIn("xpu_alignment_collect.py", collector)
        self.assertIn("GH_TOKEN: ${{ github.token }}", collector)
        self.assertNotIn("BEDROCK", collector)
        self.assertNotIn("issues: write", collector)
        self.assertLess(
            self.text.index("xpu_alignment_collect.py"),
            self.text.index("role `scan-prepare`"),
        )

    def test_agent_model_remains_the_global_opus_profile(self) -> None:
        self.assertIn("BEDROCK_MODEL: global.anthropic.claude-opus-5", self.text)
        self.assertEqual(self.text.count("--model ${{ env.BEDROCK_MODEL }}"), 3)

    def test_agents_have_enough_turns_for_the_daily_inventory(self) -> None:
        self.assertEqual(self.text.count("--max-turns 300"), 3)

    def test_reproducer_uses_the_bmg_runtime_image(self) -> None:
        prepare = self.text.split("  scan-prepare:", 1)[1].split(
            "  run-reproducers:", 1
        )[0]
        runner = self.text.split("  run-reproducers:", 1)[1].split(
            "  scan-finalize:", 1
        )[0]
        self.assertIn("intelgpu/ubuntu-26.04-rolling:26.18", prepare)
        self.assertIn("intelgpu/ubuntu-26.04-rolling:26.18", runner)
        self.assertNotIn("intelgpu/ubuntu-24.04-lts2:2523.40", runner)

    def test_prepare_restores_access_to_its_runner_mounts(self) -> None:
        prepare = self.text.split("  scan-prepare:", 1)[1].split(
            "  run-reproducers:", 1
        )[0]
        cleanup = prepare.split(
            "- name: Restore runner access after XPU preparation", 1
        )[1]
        self.assertIn("if: always()", cleanup)
        self.assertIn("stat --format='%u:%g' /__e", cleanup)
        for target in (
            '"$GITHUB_WORKSPACE"',
            "/__w/_actions",
            "/__w/_tool",
            "/__w/_temp",
            "/github",
        ):
            self.assertIn(target, cleanup)
        self.assertIn("chown --recursive --no-dereference", cleanup)
        self.assertIn("chmod --recursive u+rwX", cleanup)

    def test_every_stage_restores_the_original_collection(self) -> None:
        self.assertEqual(self.text.count("name: ${{ env.ALIGNMENT_ARTIFACT }}-collection"), 6)
        self.assertIn("--collection-root collection_snapshot", self.text)
        self.assertIn("needs: [collect, scan-prepare]", self.text)

    def test_agent_roles_surround_the_deterministic_runner(self) -> None:
        positions = [
            self.text.index("xpu_alignment_collect.py"),
            self.text.index("role `scan-prepare`"),
            self.text.index("xpu_alignment_runner.py"),
            self.text.index("role `scan-finalize`"),
            self.text.index("role `review`"),
            self.text.index("Build publishing decision"),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_agent_prompts_preserve_the_mvp_candidate_scope(self) -> None:
        prepare = self.text.split("  scan-prepare:", 1)[1].split(
            "  run-reproducers:", 1
        )[0]
        prepare = " ".join(prepare.split())
        self.assertIn("already scoped exclusively to XPU", prepare)
        self.assertIn("CPU, CUDA, ROCm, MPS", prepare)
        self.assertIn("A title, label, or XPU mention alone", prepare)
        self.assertIn("linked issue, PR, and commit chain", prepare)

        review = self.text.split("  independent-review:", 1)[1].split(
            "  gate-and-publish:", 1
        )[0]
        review = " ".join(review.split())
        self.assertIn("existing intel/torch-xpu-ops tracker", review)
        self.assertIn("do not create a payload or comment on it", review)

        finalize = self.text.split("  scan-finalize:", 1)[1].split(
            "  independent-review:", 1
        )[0]
        finalize = " ".join(finalize.split())
        self.assertIn("blocked result for that unit", finalize)
        self.assertIn("unit blocker may still allow other reviewed units to publish", finalize)

    def test_untrusted_reproducer_user_has_no_outbound_network(self) -> None:
        runner = self.text.split("  run-reproducers:", 1)[1].split(
            "  scan-finalize:", 1
        )[0]
        self.assertIn("--cap-add=NET_ADMIN", runner)
        self.assertIn("iptables --wait --append OUTPUT", runner)
        self.assertIn("ip6tables --wait --append OUTPUT", runner)
        self.assertIn('--uid-owner "$repro_uid"', runner)
        self.assertIn("Outbound network remains available to xpu-repro", runner)

    def test_reproducer_freezes_only_immutable_inputs(self) -> None:
        runner = self.text.split("  run-reproducers:", 1)[1].split(
            "  scan-finalize:", 1
        )[0]
        self.assertIn("alignment-run/prepare.json", runner)
        self.assertIn("alignment-run/scripts", runner)
        self.assertIn("alignment-run/collection", runner)
        self.assertNotIn("alignment-run/torch_compile_debug", runner)
        self.assertNotIn("export TORCH_COMPILE_DEBUG_DIR", runner)

    def test_deterministic_runner_owns_the_only_xpu_environment_probe(self) -> None:
        runner = self.text.split("  run-reproducers:", 1)[1].split(
            "  scan-finalize:", 1
        )[0]
        provision = runner.split("- name: Provision nightly XPU runtime", 1)[1].split(
            "- name: Execute as an unprivileged credential-free user", 1
        )[0]
        self.assertNotIn("torch.xpu.is_available", provision)
        self.assertNotIn("torch.xpu.get_device_name", provision)
        self.assertIn("xpu_alignment_runner.py", runner)

    def test_finalize_copies_the_deterministic_runner_environment(self) -> None:
        finalize = self.text.split("  scan-finalize:", 1)[1].split(
            "  independent-review:", 1
        )[0]
        self.assertIn("copy its environment object exactly into scan.json", finalize)

    def test_unit_blockers_do_not_fail_the_workflow_report(self) -> None:
        self.assertIn("(.global_blockers // []) | length > 0", self.text)
        self.assertIn('.collection_status == "partial"', self.text)
        self.assertIn("reviewed candidates remain publishable", self.text)

    def test_reproducer_restores_access_to_its_runner_mounts(self) -> None:
        runner = self.text.split("  run-reproducers:", 1)[1].split(
            "  scan-finalize:", 1
        )[0]
        cleanup = runner.split(
            "- name: Restore runner access after XPU execution", 1
        )[1]
        self.assertIn("if: always()", cleanup)
        self.assertIn("stat --format='%u:%g' /__e", cleanup)
        for target in (
            '"$GITHUB_WORKSPACE"',
            "/__w/_actions",
            "/__w/_tool",
            "/__w/_temp",
            "/github",
        ):
            self.assertIn(target, cleanup)
        self.assertIn("chown --recursive --no-dereference", cleanup)
        self.assertIn("chmod --recursive u+rwX", cleanup)

    def test_only_gate_job_requests_issue_write_permission(self) -> None:
        self.assertEqual(self.text.count("issues: write"), 1)
        gate = self.text.split("  gate-and-publish:", 1)[1]
        self.assertRegex(gate, re.compile(r"permissions:\n(?:.*\n){0,4}\s+issues: write"))

    def test_github_reading_agents_have_explicit_read_permissions(self) -> None:
        prepare = self.text.split("  scan-prepare:", 1)[1].split(
            "  run-reproducers:", 1
        )[0]
        review = self.text.split("  independent-review:", 1)[1].split(
            "  gate-and-publish:", 1
        )[0]
        for job in (prepare, review):
            self.assertIn("issues: read", job)
            self.assertIn("pull-requests: read", job)

    def test_schedule_and_dispatch_share_non_cancelling_concurrency(self) -> None:
        self.assertIn("group: xpu-alignment", self.text)
        self.assertIn("cancel-in-progress: false", self.text)
        self.assertIn("cron: '0 2 * * *'", self.text)

    def test_automatic_and_manual_filing_share_one_lock(self) -> None:
        gate = self.text.split("  gate-and-publish:", 1)[1]
        file_job = self.bot_text.split("  file:", 1)[1].split("  triage:", 1)[0]
        for job in (gate, file_job):
            self.assertIn("group: xpu-alignment-filing", job)
            self.assertIn("cancel-in-progress: false", job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
