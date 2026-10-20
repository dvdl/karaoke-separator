import argparse
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import endpoint
from ab import pod


class MockHTTPResponse:
    def __init__(self, status=200, data=None):
        self.status = status
        self.code = status
        if data is None:
            self._data = b""
        elif isinstance(data, (dict, list)):
            self._data = json.dumps(data).encode("utf-8")
        elif isinstance(data, str):
            self._data = data.encode("utf-8")
        else:
            self._data = bytes(data)

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestEndpointV2(unittest.TestCase):
    def setUp(self):
        self.recorded_requests = []
        self.key_patcher = patch.object(endpoint, "get_key", return_value="mock-key-for-test")
        self.key_patcher.start()
        self.addCleanup(self.key_patcher.stop)

    def mock_urlopen(self, responder=None):
        def _handler(req, timeout=60):
            method = req.get_method()
            url = req.full_url
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            headers = dict(req.headers)
            self.recorded_requests.append({
                "method": method,
                "url": url,
                "body": body,
                "headers": headers,
            })
            if responder:
                res = responder(method, url, body)
                if isinstance(res, tuple):
                    return MockHTTPResponse(status=res[0], data=res[1])
                return MockHTTPResponse(status=200, data=res)
            return MockHTTPResponse(status=200, data={})
        return patch("urllib.request.urlopen", side_effect=_handler)

    def test_constants_and_base_url(self):
        self.assertEqual(endpoint.REST_BASE, "https://api.runpod.io/v2")

    def test_unwrap(self):
        self.assertEqual(endpoint.unwrap({"endpoints": [{"id": "ep1"}]}), [{"id": "ep1"}])
        self.assertEqual(endpoint.unwrap({"workers": [{"id": "w1"}]}), [{"id": "w1"}])
        self.assertEqual(endpoint.unwrap({"templates": [{"id": "t1"}]}), [{"id": "t1"}])
        self.assertEqual(endpoint.unwrap({"pods": [{"id": "p1"}]}), [{"id": "p1"}])
        self.assertEqual(endpoint.unwrap({"other": "value"}), {"other": "value"})
        self.assertEqual(endpoint.unwrap([1, 2, 3]), [1, 2, 3])
        self.assertEqual(endpoint.unwrap("string"), "string")

    def test_create(self):
        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/templates":
                return 201, {"id": "tmpl-test-123"}
            if url == "https://api.runpod.io/v2/serverless":
                return 201, {"id": "ep-test-456"}
            if url.startswith("https://api.runpod.io/v2/serverless/"):
                return 200, {"id": "ep-test-456"}
            return 200, {}

        fake_idfile = MagicMock()
        fake_idfile.exists.return_value = False

        with patch.object(endpoint, "IDFILE", fake_idfile), self.mock_urlopen(responder):
            endpoint.create()

        # Check recorded requests:
        # 1: POST /v2/templates
        # 2: POST /v2/serverless
        # 3: PATCH /v2/serverless/{id} (via gpus())
        self.assertEqual(len(self.recorded_requests), 3)

        req1 = self.recorded_requests[0]
        self.assertEqual(req1["method"], "POST")
        self.assertEqual(req1["url"], "https://api.runpod.io/v2/templates")
        self.assertEqual(req1["body"]["name"], "karaoke-separator-deux")
        self.assertIn("image", req1["body"])
        self.assertNotIn("imageName", req1["body"])
        self.assertIn("disk", req1["body"])
        self.assertNotIn("containerDiskInGb", req1["body"])
        self.assertEqual(req1["body"]["disk"], 15)
        self.assertEqual(req1["body"]["serverless"], True)
        self.assertNotIn("isServerless", req1["body"])
        self.assertEqual(req1["body"]["startJupyter"], False)

        req2 = self.recorded_requests[1]
        self.assertEqual(req2["method"], "POST")
        self.assertEqual(req2["url"], "https://api.runpod.io/v2/serverless")
        self.assertEqual(req2["body"]["name"], "karaoke-separator-deux")
        self.assertEqual(req2["body"]["type"], "QUEUE")
        self.assertEqual(req2["body"]["templateId"], "tmpl-test-123")
        self.assertIn("gpu", req2["body"])
        self.assertEqual(req2["body"]["gpu"]["pools"], endpoint.POOLS)
        self.assertEqual(req2["body"]["gpu"]["excludedTypes"], endpoint.EXCLUDE)
        self.assertEqual(req2["body"]["gpu"]["count"], 1)
        self.assertEqual(req2["body"]["gpu"]["minCudaVersion"], "12.8")
        self.assertEqual(req2["body"]["workers"], {"min": 0, "max": 1, "idleTimeout": 5})
        self.assertEqual(req2["body"]["scaling"], {"type": "QUEUE_DELAY", "queueDelay": 4})
        self.assertEqual(req2["body"]["timeout"], 300000)
        self.assertEqual(req2["body"]["flashboot"], "FLASHBOOT")

        req3 = self.recorded_requests[2]
        self.assertEqual(req3["method"], "PATCH")
        self.assertTrue(req3["url"].startswith("https://api.runpod.io/v2/serverless/"))
        self.assertIn("gpu", req3["body"])
        self.assertEqual(req3["body"]["gpu"]["pools"], endpoint.POOLS)
        self.assertEqual(req3["body"]["gpu"]["excludedTypes"], endpoint.EXCLUDE)
        self.assertEqual(req3["body"]["scaling"], {"type": "QUEUE_DELAY", "queueDelay": 4})
        self.assertEqual(req3["body"]["workers"], {"min": 0, "max": 1, "idleTimeout": 5})
        self.assertEqual(req3["body"]["flashboot"], "FLASHBOOT")

        fake_idfile.write_text.assert_called_once_with("ep-test-456 tmpl-test-123\n")

    def test_status(self):
        fake_idfile = MagicMock()
        fake_idfile.read_text.return_value = "ep-test-123 tmpl-123\n"

        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/serverless/ep-test-123":
                return 200, {
                    "id": "ep-test-123",
                    "workers": {"min": 0, "max": 1, "idleTimeout": 5},
                    "timeout": 300000,
                }
            if url == "https://api.runpod.io/v2/serverless/ep-test-123/workers":
                return 200, {
                    "workers": [{"id": "w1", "status": "RUNNING"}]
                }
            if url == "https://api.runpod.ai/v2/ep-test-123/health":
                return 200, {"workers": {"ready": 1}}
            return 200, {}

        stdout = io.StringIO()
        with patch.object(endpoint, "IDFILE", fake_idfile), self.mock_urlopen(responder), patch("sys.stdout", stdout):
            endpoint.status()

        self.assertEqual(len(self.recorded_requests), 3)
        urls = [r["url"] for r in self.recorded_requests]
        self.assertIn("https://api.runpod.io/v2/serverless/ep-test-123", urls)
        self.assertIn("https://api.runpod.io/v2/serverless/ep-test-123/workers", urls)
        self.assertIn("https://api.runpod.ai/v2/ep-test-123/health", urls)
        output = stdout.getvalue()
        self.assertIn("w1", output)
        self.assertIn("RUNNING", output)
        self.assertIn("300000", output)

    def test_status_rfc9457_error(self):
        fake_idfile = MagicMock()
        fake_idfile.read_text.return_value = "ep-test-123 tmpl-123\n"

        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/serverless/ep-test-123":
                return 404, {"title": "Not Found", "status": 404, "detail": "endpoint ep-test-123 not found"}
            if url == "https://api.runpod.io/v2/serverless/ep-test-123/workers":
                return 404, {"title": "Not Found", "status": 404, "detail": "workers not found"}
            if url == "https://api.runpod.ai/v2/ep-test-123/health":
                return 500, "internal error"
            return 200, {}

        stdout = io.StringIO()
        with patch.object(endpoint, "IDFILE", fake_idfile), self.mock_urlopen(responder), patch("sys.stdout", stdout):
            endpoint.status()

        output = stdout.getvalue()
        self.assertIn("[404] Not Found - endpoint ep-test-123 not found", output)
        self.assertIn("[404] Not Found - workers not found", output)

    def test_format_error(self):
        self.assertEqual(
            endpoint.format_error({"title": "Not Found", "status": 404, "detail": "ep not found"}),
            "[404] Not Found - ep not found",
        )
        self.assertEqual(
            endpoint.format_error({"title": "Bad Request", "status": 400, "detail": "invalid input", "errors": ["err1", "err2"]}),
            "[400] Bad Request - invalid input (err1; err2)",
        )
        self.assertEqual(endpoint.format_error("raw string error"), "raw string error")

    def test_missing_env_file_fails_loudly(self):
        self.key_patcher.stop()
        fake_env = Path("/tmp/nonexistent_env_file_test_endpoint")
        with patch.object(endpoint, "ENV_FILE", fake_env):
            with self.assertRaises(FileNotFoundError):
                endpoint.get_key()

    def test_setmax(self):
        fake_idfile = MagicMock()
        fake_idfile.read_text.return_value = "ep-test-123 tmpl-123\n"

        with patch.object(endpoint, "IDFILE", fake_idfile), self.mock_urlopen():
            endpoint.setmax(0)
            endpoint.setmax(1)

        self.assertEqual(len(self.recorded_requests), 2)
        req0 = self.recorded_requests[0]
        self.assertEqual(req0["method"], "PATCH")
        self.assertEqual(req0["url"], "https://api.runpod.io/v2/serverless/ep-test-123")
        self.assertEqual(req0["body"], {"workers": {"max": 0}})

        req1 = self.recorded_requests[1]
        self.assertEqual(req1["method"], "PATCH")
        self.assertEqual(req1["url"], "https://api.runpod.io/v2/serverless/ep-test-123")
        self.assertEqual(req1["body"], {"workers": {"max": 1}})

    def test_gpus_no_graphql(self):
        fake_idfile = MagicMock()
        fake_idfile.read_text.return_value = "ep-test-123 tmpl-123\n"

        with patch.object(endpoint, "IDFILE", fake_idfile), self.mock_urlopen():
            endpoint.gpus()

        self.assertEqual(len(self.recorded_requests), 1)
        req = self.recorded_requests[0]
        self.assertEqual(req["method"], "PATCH")
        self.assertEqual(req["url"], "https://api.runpod.io/v2/serverless/ep-test-123")
        self.assertNotIn("graphql", req["url"])
        self.assertIn("gpu", req["body"])
        self.assertEqual(req["body"]["gpu"]["pools"], endpoint.POOLS)
        self.assertEqual(req["body"]["gpu"]["excludedTypes"], endpoint.EXCLUDE)
        self.assertEqual(req["body"]["gpu"]["count"], 1)
        self.assertEqual(req["body"]["scaling"], {"type": "QUEUE_DELAY", "queueDelay": 4})
        self.assertEqual(req["body"]["workers"], {"min": 0, "max": 1, "idleTimeout": 5})
        self.assertEqual(req["body"]["flashboot"], "FLASHBOOT")


class TestPodV2(unittest.TestCase):
    def setUp(self):
        self.recorded_requests = []
        self.key_patcher = patch.object(pod, "get_key", return_value="mock-key-for-test")
        self.key_patcher.start()
        self.addCleanup(self.key_patcher.stop)
        self.pubkey_patcher = patch.object(pod, "get_pubkey", return_value="ssh-rsa test-pubkey-maaravi")
        self.pubkey_patcher.start()
        self.addCleanup(self.pubkey_patcher.stop)

    def mock_urlopen(self, responder=None):
        def _handler(req, timeout=60):
            method = req.get_method()
            url = req.full_url
            body = json.loads(req.data.decode("utf-8")) if req.data else None
            headers = dict(req.headers)
            self.recorded_requests.append({
                "method": method,
                "url": url,
                "body": body,
                "headers": headers,
            })
            if responder:
                res = responder(method, url, body)
                if isinstance(res, tuple):
                    return MockHTTPResponse(status=res[0], data=res[1])
                return MockHTTPResponse(status=200, data=res)
            return MockHTTPResponse(status=200, data={})
        return patch("urllib.request.urlopen", side_effect=_handler)

    def test_constants_and_base_url(self):
        self.assertEqual(pod.REST, "https://api.runpod.io/v2")

    def test_unwrap(self):
        self.assertEqual(pod.unwrap({"pods": [{"id": "p1"}]}), [{"id": "p1"}])
        self.assertEqual(pod.unwrap({"endpoints": [{"id": "e1"}]}), [{"id": "e1"}])
        self.assertEqual(pod.unwrap({"workers": [{"id": "w1"}]}), [{"id": "w1"}])
        self.assertEqual(pod.unwrap({"templates": [{"id": "t1"}]}), [{"id": "t1"}])
        self.assertEqual(pod.unwrap({"other": "value"}), {"other": "value"})
        self.assertEqual(pod.unwrap([1, 2]), [1, 2])

    def test_live_ids_unwrapping(self):
        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/pods" and method == "GET":
                return 200, {"pods": [{"id": "pod-a"}, {"id": "pod-b"}]}
            return 200, {}

        with self.mock_urlopen(responder):
            ids = pod.live_ids()

        self.assertEqual(ids, ["pod-a", "pod-b"])
        self.assertEqual(len(self.recorded_requests), 1)
        self.assertEqual(self.recorded_requests[0]["method"], "GET")
        self.assertEqual(self.recorded_requests[0]["url"], "https://api.runpod.io/v2/pods")

    def test_live_ids_error(self):
        def responder(method, url, body):
            return 500, "error"

        with self.mock_urlopen(responder):
            ids = pod.live_ids()
        self.assertIsNone(ids)

    def test_cmd_up_nested_body(self):
        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/pods" and method == "POST":
                return 201, {"id": "pod-created-999"}
            return 200, {}

        fake_podfile = MagicMock()
        fake_podfile.exists.return_value = False

        args = argparse.Namespace(disk=40, max_minutes=60)
        with patch.object(pod, "PODFILE", fake_podfile), self.mock_urlopen(responder):
            pod.cmd_up(args)

        self.assertEqual(len(self.recorded_requests), 1)
        req = self.recorded_requests[0]
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["url"], "https://api.runpod.io/v2/pods")

        body = req["body"]
        self.assertEqual(body["name"], pod.NAME)
        self.assertEqual(body["image"], pod.IMAGE)
        self.assertNotIn("imageName", body)
        self.assertEqual(body["gpu"], {"id": pod.GPU_CANDIDATES[0][0], "count": 1})
        self.assertNotIn("gpuTypeIds", body)
        self.assertNotIn("gpuCount", body)
        self.assertEqual(body["cloud"], pod.GPU_CANDIDATES[0][1])
        self.assertNotIn("cloudType", body)
        self.assertEqual(body["disk"], 40)
        self.assertNotIn("containerDiskInGb", body)
        self.assertNotIn("volumeInGb", body)
        self.assertEqual(body["ports"], ["22/tcp"])
        self.assertEqual(body["env"]["MAX_MINUTES"], "60")
        self.assertEqual(body["env"]["PUBLIC_KEY"], "ssh-rsa test-pubkey-maaravi")

        fake_podfile.write_text.assert_called_once()
        saved_text = fake_podfile.write_text.call_args[0][0]
        self.assertTrue(saved_text.startswith("pod-created-999\n"))

    def test_cmd_status(self):
        fake_podfile = MagicMock()
        fake_podfile.exists.return_value = True
        fake_podfile.read_text.return_value = "pod-123\nNVIDIA RTX A5000\nSECURE\n0.27\n1700000000.0\n"

        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/pods/pod-123":
                return 200, {"id": "pod-123", "status": "RUNNING"}
            return 200, {}

        stdout = io.StringIO()
        with patch.object(pod, "PODFILE", fake_podfile), self.mock_urlopen(responder), patch("sys.stdout", stdout):
            pod.cmd_status(argparse.Namespace())

        self.assertEqual(len(self.recorded_requests), 1)
        self.assertEqual(self.recorded_requests[0]["method"], "GET")
        self.assertEqual(self.recorded_requests[0]["url"], "https://api.runpod.io/v2/pods/pod-123")
        self.assertIn("status=RUNNING", stdout.getvalue())

    def test_cmd_status_rfc9457_error(self):
        fake_podfile = MagicMock()
        fake_podfile.exists.return_value = True
        fake_podfile.read_text.return_value = "pod-123\nNVIDIA RTX A5000\nSECURE\n0.27\n1700000000.0\n"

        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/pods/pod-123":
                return 404, {"title": "Not Found", "status": 404, "detail": "pod not found"}
            return 200, {}

        stdout = io.StringIO()
        with patch.object(pod, "PODFILE", fake_podfile), self.mock_urlopen(responder), patch("sys.stdout", stdout):
            pod.cmd_status(argparse.Namespace())

        self.assertIn("status=[404] Not Found - pod not found", stdout.getvalue())

    def test_cmd_down_delete(self):
        fake_podfile = MagicMock()
        fake_podfile.exists.return_value = True
        fake_podfile.read_text.return_value = "pod-123\nNVIDIA RTX A5000\nSECURE\n0.27\n1700000000.0\n"

        call_count = {"live": 0}

        def responder(method, url, body):
            if method == "DELETE" and url == "https://api.runpod.io/v2/pods/pod-123":
                return 204, {}
            if method == "GET" and url == "https://api.runpod.io/v2/pods":
                call_count["live"] += 1
                return 200, {"pods": []}
            return 200, {}

        with patch.object(pod, "PODFILE", fake_podfile), self.mock_urlopen(responder), patch("time.sleep"):
            pod.cmd_down(argparse.Namespace(pod_id=None))

        delete_reqs = [r for r in self.recorded_requests if r["method"] == "DELETE"]
        self.assertTrue(len(delete_reqs) >= 1)
        self.assertEqual(delete_reqs[0]["url"], "https://api.runpod.io/v2/pods/pod-123")
        fake_podfile.unlink.assert_called_once()

    def test_cmd_sshcfg_v2(self):
        fake_podfile = MagicMock()
        fake_podfile.exists.return_value = True
        fake_podfile.read_text.return_value = "pod-123\nNVIDIA RTX A5000\nSECURE\n0.27\n1700000000.0\n"

        def responder(method, url, body):
            if url == "https://api.runpod.io/v2/pods/pod-123":
                return 200, {
                    "id": "pod-123",
                    "status": "RUNNING",
                    "ssh": {
                        "direct": {
                            "host": "198.51.100.25",
                            "port": 22022,
                            "username": "root",
                            "command": "ssh root@198.51.100.25 -p 22022",
                        }
                    }
                }
            return 200, {}

        fake_cfg = MagicMock()
        with patch.object(pod, "PODFILE", fake_podfile), self.mock_urlopen(responder), \
             patch.object(pod, "ROOT", Path("/tmp")), \
             patch("subprocess.run") as mock_subproc, \
             patch("pathlib.Path.write_text") as mock_write, \
             patch("pathlib.Path.mkdir"):
            mock_subproc.return_value.returncode = 0
            pod.cmd_sshcfg(argparse.Namespace(pod_id=None))

        mock_write.assert_called_once()
        written = mock_write.call_args[0][0]
        self.assertIn("HostName 198.51.100.25", written)
        self.assertIn("Port 22022", written)

    def test_pod_actions(self):
        with self.mock_urlopen():
            pod.action("pod-xyz", "restart")
            pod.stop_pod("pod-xyz")
            pod.start_pod("pod-xyz")

        self.assertEqual(len(self.recorded_requests), 3)
        self.assertEqual(self.recorded_requests[0]["method"], "POST")
        self.assertEqual(self.recorded_requests[0]["url"], "https://api.runpod.io/v2/pods/pod-xyz/action")
        self.assertEqual(self.recorded_requests[0]["body"], {"action": "restart"})

        self.assertEqual(self.recorded_requests[1]["method"], "POST")
        self.assertEqual(self.recorded_requests[1]["url"], "https://api.runpod.io/v2/pods/pod-xyz/action")
        self.assertEqual(self.recorded_requests[1]["body"], {"action": "stop"})

        self.assertEqual(self.recorded_requests[2]["method"], "POST")
        self.assertEqual(self.recorded_requests[2]["url"], "https://api.runpod.io/v2/pods/pod-xyz/action")
        self.assertEqual(self.recorded_requests[2]["body"], {"action": "start"})

    def test_missing_files_fail_loudly(self):
        self.key_patcher.stop()
        self.pubkey_patcher.stop()
        fake_env = Path("/tmp/nonexistent_env_file_test_pod")
        fake_pubkey = Path("/tmp/nonexistent_pubkey_test_pod")
        with patch.object(pod, "ENV_FILE", fake_env):
            with self.assertRaises(FileNotFoundError):
                pod.get_key()
        with patch.object(pod, "PUBKEY_FILE", fake_pubkey):
            with self.assertRaises(FileNotFoundError):
                pod.get_pubkey()


class TestOpenAPIValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec_path = Path("/tmp/runpod_v2_openapi.json")
        if not spec_path.exists():
            raise unittest.SkipTest("/tmp/runpod_v2_openapi.json not found")
        with open(spec_path) as f:
            cls.spec = json.load(f)
        cls.schemas = cls.spec["components"]["schemas"]

    @classmethod
    def _get_schema_properties(cls, s):
        props = {}
        if "$ref" in s:
            ref_name = s["$ref"].split("/")[-1]
            props.update(cls._get_schema_properties(cls.schemas[ref_name]))
        if "allOf" in s:
            for sub in s["allOf"]:
                props.update(cls._get_schema_properties(sub))
        if "properties" in s:
            props.update(s["properties"])
        return props

    def assertValidSchema(self, schema_name, body):
        schema = self.schemas[schema_name]
        allowed_props = self._get_schema_properties(schema)
        for k in body:
            self.assertIn(k, allowed_props, f"Key {k!r} not in {schema_name}")

    def test_create_template_request(self):
        body = {
            "name": "karaoke-separator-deux",
            "image": endpoint.IMAGE,
            "serverless": True,
            "startJupyter": False,
            "disk": 15,
            "env": {},
        }
        self.assertValidSchema("CreateTemplateRequest", body)

    def test_create_endpoint_request(self):
        body = {
            "name": "karaoke-separator-deux",
            "type": "QUEUE",
            "templateId": "test-template-id",
            "gpu": {"pools": endpoint.POOLS, "excludedTypes": endpoint.EXCLUDE, "count": 1, "minCudaVersion": "12.8"},
            "workers": {"min": 0, "max": 1, "idleTimeout": 5},
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
            "timeout": 300000,
            "flashboot": "FLASHBOOT",
        }
        self.assertValidSchema("CreateEndpointRequest", body)

    def test_update_endpoint_request(self):
        body_gpus = {
            "gpu": {"pools": endpoint.POOLS, "excludedTypes": endpoint.EXCLUDE, "count": 1},
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
            "workers": {"min": 0, "max": 1, "idleTimeout": 5},
            "flashboot": "FLASHBOOT",
        }
        self.assertValidSchema("UpdateEndpointRequest", body_gpus)

        body_setmax = {
            "workers": {"max": 0},
        }
        self.assertValidSchema("UpdateEndpointRequest", body_setmax)

    def test_create_pod_request(self):
        body = {
            "name": pod.NAME,
            "image": pod.IMAGE,
            "gpu": {"id": pod.GPU_CANDIDATES[0][0], "count": 1},
            "cloud": pod.GPU_CANDIDATES[0][1],
            "disk": 30,
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": "ssh-rsa mock", "MAX_MINUTES": "45"},
        }
        self.assertValidSchema("CreatePodRequest", body)

    def test_pod_action_request(self):
        for act in ("stop", "start", "restart"):
            body = {"action": act}
            self.assertValidSchema("PodActionRequest", body)


if __name__ == "__main__":
    unittest.main()
