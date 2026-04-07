import sys
import unittest
from pathlib import Path

from starlette.websockets import WebSocketDisconnect

# Ensure `src/` is on PYTHONPATH so `import vibevoice` works in tests.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class TestRealtimeWebSocket(unittest.TestCase):
    def setUp(self) -> None:
        # Import inside setUp so we can patch config globals per-test.
        from vibevoice.config import config

        # Require an API key for these tests.
        config.API_KEY = "test-key"

        from vibevoice.main import app
        from fastapi.testclient import TestClient

        self.client = TestClient(app)

    def test_realtime_ws_requires_api_key(self) -> None:
        # No key should be rejected (policy violation 1008).
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/api/v1/speech/realtime") as ws:
                # Server should close immediately; any read triggers disconnect.
                ws.receive_text()

    def test_realtime_ws_happy_path_buffered_flush(self) -> None:
        with self.client.websocket_connect(
            "/api/v1/speech/realtime?api_key=test-key"
        ) as ws:
            # Initial status frame
            msg = ws.receive_text()
            self.assertIn('"type": "status"', msg)

            # Start session
            ws.send_text('{"type":"start","cfg_scale":1.5,"inference_steps":5}')
            msg = ws.receive_text()
            self.assertIn("session_started", msg)

            # Flush with no buffered text should return an error (and not require upstream).
            ws.send_text('{"type":"flush"}')
            msg = ws.receive_text()
            self.assertIn('"type": "error"', msg)


if __name__ == "__main__":
    unittest.main()

