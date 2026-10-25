import base64
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULES_TO_STUB = [
    "torch",
    "torch.cuda",
    "runpod",
    "runpod.serverless",
    "audio_separator",
    "audio_separator.separator",
    "gpucheck",
    "soundfile",
]

mock_torch = None
mock_runpod = None
mock_audio_sep = None
mock_gpucheck = None
mock_sf = None
handler = None


def setUpModule():
    global mock_torch, mock_runpod, mock_audio_sep, mock_gpucheck, mock_sf, handler

    # 1. Stub torch
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.get_device_name.return_value = "cpu"
    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_torch.cuda

    # 2. Stub runpod
    mock_runpod = MagicMock()
    sys.modules["runpod"] = mock_runpod
    sys.modules["runpod.serverless"] = mock_runpod.serverless

    # 3. Stub audio_separator
    mock_audio_sep = MagicMock()
    mock_separator_mod = MagicMock()
    mock_separator_cls = MagicMock()
    mock_separator_mod.Separator = mock_separator_cls
    mock_audio_sep.separator = mock_separator_mod
    sys.modules["audio_separator"] = mock_audio_sep
    sys.modules["audio_separator.separator"] = mock_separator_mod

    # 4. Stub gpucheck
    mock_gpucheck = MagicMock()
    mock_gpucheck.is_mig = MagicMock(return_value=False)
    sys.modules["gpucheck"] = mock_gpucheck

    # 5. Stub soundfile
    mock_sf = MagicMock()
    mock_info = MagicMock()
    mock_info.duration = 60.0
    mock_sf.info.return_value = mock_info
    mock_sf.read.return_value = (MagicMock(), 44100)
    mock_sf.write.return_value = None
    sys.modules["soundfile"] = mock_sf

    # Import handler after sys.modules are stubbed
    import handler as _handler
    handler = _handler


def tearDownModule():
    global handler
    for m in MODULES_TO_STUB:
        sys.modules.pop(m, None)
    sys.modules.pop("handler", None)
    handler = None


class TestVocsplitTask(unittest.TestCase):
    def setUp(self):
        # Reset vocsplit global state between tests
        handler.VOCSPLIT_SEP = None
        mock_gpucheck.is_mig.return_value = False
        mock_sf.reset_mock()

    def test_no_task_calls_old_path(self):
        """(1) no task or task == 'separate' calls the old separate path."""
        # 1a. Input without "task"
        with patch.object(handler, "_handle_separate", return_value={"mock": "old_path"}) as mock_old:
            job = {"input": {"audio_b64": "chunk_data"}}
            res = handler.handler(job)
            mock_old.assert_called_once_with({"audio_b64": "chunk_data"})
            self.assertEqual(res, {"mock": "old_path"})

        # 1b. Input with task == "separate"
        with patch.object(handler, "_handle_separate", return_value={"mock": "separate_path"}) as mock_old:
            job = {"input": {"task": "separate", "audio_b64": "chunk_data"}}
            res = handler.handler(job)
            mock_old.assert_called_once_with({"task": "separate", "audio_b64": "chunk_data"})
            self.assertEqual(res, {"mock": "separate_path"})

    def test_vocsplit_lead_backing_mapping(self):
        """(2) task vocsplit -> lead/backing mapping from fake Separator file names."""
        lead_raw = b"PCM24_FLAC_LEAD_AUDIO_DATA"
        backing_raw = b"PCM24_FLAC_BACKING_AUDIO_DATA"

        fake_sep = MagicMock()

        def fake_separate(src):
            lead_file = os.path.join(handler.VOCSPLIT_OUT, "x_(Vocals)_chunk0.flac")
            backing_file = os.path.join(handler.VOCSPLIT_OUT, "x_(Instrumental)_chunk0.flac")
            with open(lead_file, "wb") as f:
                f.write(lead_raw)
            with open(backing_file, "wb") as f:
                f.write(backing_raw)
            return [lead_file, backing_file]

        fake_sep.separate.side_effect = fake_separate

        with patch.object(handler, "get_vocsplit_separator", return_value=(fake_sep, 2.5)):
            job = {
                "input": {
                    "task": "vocsplit",
                    "audio_b64": base64.b64encode(b"input_audio").decode("utf-8"),
                }
            }
            res = handler.handler(job)

            self.assertEqual(res.get("task"), "vocsplit")
            self.assertEqual(res.get("model"), "becruily-karaoke")
            self.assertEqual(res.get("gpu"), handler.GPU)
            self.assertEqual(res.get("audio_seconds"), 60.0)
            self.assertIn("separate_seconds", res)
            self.assertEqual(res.get("load_seconds"), 2.5)

            # Check lead and backing base64 encoding matches the corresponding files
            expected_lead = base64.b64encode(lead_raw).decode("utf-8")
            expected_backing = base64.b64encode(backing_raw).decode("utf-8")
            self.assertEqual(res.get("lead_b64"), expected_lead)
            self.assertEqual(res.get("backing_b64"), expected_backing)

            # Verify soundfile was asked to write FLAC PCM_24
            calls = mock_sf.write.call_args_list
            self.assertEqual(len(calls), 2)
            for c in calls:
                self.assertEqual(c.kwargs.get("format"), "FLAC")
                self.assertEqual(c.kwargs.get("subtype"), "PCM_24")

    def test_vocsplit_reversed_filenames_mapping(self):
        """Lead/backing mapping works regardless of the order Separator returns files in."""
        lead_raw = b"LEAD_BYTES"
        backing_raw = b"BACKING_BYTES"

        fake_sep = MagicMock()

        def fake_separate_reversed(src):
            lead_file = os.path.join(handler.VOCSPLIT_OUT, "track_(Vocals)_stem.flac")
            backing_file = os.path.join(handler.VOCSPLIT_OUT, "track_(Instrumental)_stem.flac")
            with open(lead_file, "wb") as f:
                f.write(lead_raw)
            with open(backing_file, "wb") as f:
                f.write(backing_raw)
            # Return instrumental first, vocals second
            return [backing_file, lead_file]

        fake_sep.separate.side_effect = fake_separate_reversed

        with patch.object(handler, "get_vocsplit_separator", return_value=(fake_sep, 0)):
            job = {
                "input": {
                    "task": "vocsplit",
                    "audio_b64": base64.b64encode(b"test").decode("utf-8"),
                }
            }
            res = handler.handler(job)
            self.assertEqual(res.get("lead_b64"), base64.b64encode(lead_raw).decode("utf-8"))
            self.assertEqual(res.get("backing_b64"), base64.b64encode(backing_raw).decode("utf-8"))
            self.assertEqual(res.get("load_seconds"), 0)

    def test_unknown_task_error(self):
        """(3) unknown task -> error."""
        job = {"input": {"task": "some_random_task"}}
        res = handler.handler(job)
        self.assertIn("error", res)
        self.assertEqual(res["error"], "unknown task: some_random_task")

    def test_mig_refusal_vocsplit(self):
        """MIG GPU refusal applies to vocsplit jobs as well."""
        mock_gpucheck.is_mig.return_value = True
        job = {"input": {"task": "vocsplit", "audio_b64": "dummy"}}
        res = handler.handler(job)
        self.assertIn("error", res)
        self.assertTrue(res["error"].startswith("refused: MIG GPU"))
        self.assertEqual(res.get("refused_gpu"), handler.GPU)

    def test_sep_none_does_not_affect_vocsplit(self):
        """If main SEP is None for non-MIG reason, vocsplit must still succeed."""
        handler.SEP = None
        fake_sep = MagicMock()

        def fake_separate(src):
            lead_file = os.path.join(handler.VOCSPLIT_OUT, "x_(Vocals).flac")
            backing_file = os.path.join(handler.VOCSPLIT_OUT, "x_(Instrumental).flac")
            with open(lead_file, "wb") as f:
                f.write(b"LEAD")
            with open(backing_file, "wb") as f:
                f.write(b"BACKING")
            return [lead_file, backing_file]

        fake_sep.separate.side_effect = fake_separate

        with patch.object(handler, "get_vocsplit_separator", return_value=(fake_sep, 0)):
            job = {
                "input": {
                    "task": "vocsplit",
                    "audio_b64": base64.b64encode(b"test").decode("utf-8"),
                }
            }
            res = handler.handler(job)
            self.assertEqual(res.get("task"), "vocsplit")
            self.assertNotIn("error", res)

        # Meanwhile, default separate path returns error when SEP is None
        dummy_b64 = base64.b64encode(b"dummy_audio").decode("utf-8")
        sep_res = handler.handler({"input": {"audio_b64": dummy_b64}})
        self.assertEqual(sep_res, {"error": "separator not loaded"})

    def test_vocsplit_missing_audio(self):
        """Vocsplit without audio_url or audio_b64 returns error."""
        job = {"input": {"task": "vocsplit"}}
        res = handler.handler(job)
        self.assertEqual(res, {"error": "need audio_url or audio_b64"})

    def test_vocsplit_unexpected_stems(self):
        """Fake separator output missing required stem returns unexpected stems error."""
        fake_sep = MagicMock()
        fake_sep.separate.return_value = ["track_(Bass).flac", "track_(Drums).flac"]

        with patch.object(handler, "get_vocsplit_separator", return_value=(fake_sep, 0)):
            job = {
                "input": {
                    "task": "vocsplit",
                    "audio_b64": base64.b64encode(b"test").decode("utf-8"),
                }
            }
            res = handler.handler(job)
            self.assertIn("error", res)
            self.assertTrue(res["error"].startswith("unexpected stems"))

    def test_map_vocsplit_stems_helper(self):
        """map_vocsplit_stems helper correctly identifies lead and backing stems."""
        files = ["chunk_(Vocals)_1.flac", "chunk_(Instrumental)_1.flac"]
        lead, backing = handler.map_vocsplit_stems(files, out_dir="/tmp/test_dir")
        self.assertEqual(lead, "/tmp/test_dir/chunk_(Vocals)_1.flac")
        self.assertEqual(backing, "/tmp/test_dir/chunk_(Instrumental)_1.flac")

        # Absolute paths preserved
        abs_files = ["/opt/out/chunk_(Vocals).flac", "/opt/out/chunk_(Instrumental).flac"]
        lead, backing = handler.map_vocsplit_stems(abs_files, out_dir="/tmp/test_dir")
        self.assertEqual(lead, "/opt/out/chunk_(Vocals).flac")
        self.assertEqual(backing, "/opt/out/chunk_(Instrumental).flac")

        # Missing one of the stems
        lead, backing = handler.map_vocsplit_stems(["chunk_(Vocals)_1.flac"], out_dir="/tmp/test_dir")
        self.assertIsNotNone(lead)
        self.assertIsNone(backing)

    def test_lazy_loading_caching(self):
        """get_vocsplit_separator caches instance in VOCSPLIT_SEP and returns 0 load_seconds on subsequent calls."""
        fake_instance = MagicMock()
        handler.VOCSPLIT_SEP = fake_instance

        sep, load_sec = handler.get_vocsplit_separator()
        self.assertIs(sep, fake_instance)
        self.assertEqual(load_sec, 0)


if __name__ == "__main__":
    unittest.main()
