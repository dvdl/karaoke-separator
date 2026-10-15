import sys
import unittest
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpucheck import is_mig


class TestGpuCheck(unittest.TestCase):
    def test_no_heavy_imports(self):
        """Verifies is_mig does not import torch, runpod, or audio_separator."""
        for mod in ("torch", "runpod", "audio_separator"):
            self.assertNotIn(mod, sys.modules)

    def test_is_mig_positive(self):
        mig_devices = [
            "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
            "NVIDIA A100-SXM4-40GB MIG 3g.20gb",
            "NVIDIA H100 PCIe MIG 3g.40gb",
            "NVIDIA A30 MIG 1g.6gb",
            "MIG 1g.24gb",
            "mig 1g.24gb",
            "MIG",
            "mig",
            "Server Edition mig-slice",
        ]
        for name in mig_devices:
            with self.subTest(name=name):
                self.assertTrue(is_mig(name), f"Expected True for {name}")

    def test_is_mig_negative(self):
        non_mig_devices = [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX A5000",
            "NVIDIA RTX A4500",
            "NVIDIA L4",
            "NVIDIA RTX 2000 Ada Generation",
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA RTX A4000",
            "Tesla V100-SXM2-16GB",
            "cpu",
            "",
            None,
            "Amiga",
            "pyramid",
            "immigrant",
        ]
        for name in non_mig_devices:
            with self.subTest(name=name):
                self.assertFalse(is_mig(name), f"Expected False for {name}")


if __name__ == "__main__":
    unittest.main()
