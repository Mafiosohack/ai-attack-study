"""AttackRunner batching: same counting rules as sequential, path recorded.
Pure (no GPU): fake pipelines stand in for the model."""

from __future__ import annotations

import json
import tempfile
import unittest

from harness.runner import Attack, AttackRunner


class _Seq:
    """Sequential-only pipeline (no run_batch)."""
    def run(self, payload):
        return "HIT"

    def describe_config(self):
        return {"victim": "seq"}


class _Batch(_Seq):
    """Returns HIT/miss alternately so counts are checkable; records chunks."""
    def __init__(self, fail_chunk: int | None = None):
        self.chunks: list[int] = []
        self.fail_chunk = fail_chunk
        self.i = 0

    def run_batch(self, payload, n):
        self.chunks.append(n)
        if self.fail_chunk is not None and len(self.chunks) - 1 == self.fail_chunk:
            raise RuntimeError("boom")
        out = []
        for _ in range(n):
            out.append("HIT" if self.i % 2 == 0 else "miss")
            self.i += 1
        return out


def _attack():
    return Attack(name="t", payload=None, succeeded=lambda o: o == "HIT")


class RunnerBatching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runner = AttackRunner(results_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sequential_default_records_path(self):
        r = self.runner.measure(_Seq(), _attack(), 3)
        self.assertEqual((r.successes, r.n_trials), (3, 3))
        self.assertEqual(r.victim_config["trial_execution"], {"mode": "sequential"})

    def test_batched_chunks_and_counts(self):
        pipe = _Batch()
        r = self.runner.measure(pipe, _attack(), 20, batch_size=8)
        self.assertEqual(pipe.chunks, [8, 8, 4])
        self.assertEqual(len(r.outputs), 20)
        self.assertEqual(r.successes, 10)
        self.assertEqual(r.victim_config["trial_execution"]["mode"], "batched")
        self.assertEqual(r.victim_config["trial_execution"]["batch_size"], 8)

    def test_crashed_batch_counts_as_failures(self):
        r = self.runner.measure(_Batch(fail_chunk=1), _attack(), 20, batch_size=10)
        self.assertEqual(len(r.outputs), 20)
        self.assertEqual(r.outputs[10:], [""] * 10)
        self.assertEqual(r.successes, 5)

    def test_batch_requires_run_batch(self):
        with self.assertRaises(TypeError):
            self.runner.measure(_Seq(), _attack(), 4, batch_size=2)

    def test_saved_json_carries_execution(self):
        r = self.runner.measure(_Batch(), _attack(), 4, batch_size=4)
        files = list(__import__("pathlib").Path(self.tmp.name).glob("*.json"))
        saved = json.loads(files[0].read_text())
        self.assertEqual(saved["victim_config"]["trial_execution"]["mode"], "batched")


if __name__ == "__main__":
    unittest.main()
