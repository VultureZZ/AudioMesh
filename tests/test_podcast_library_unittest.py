import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Ensure `src/` is on PYTHONPATH so `import vibevoice` works in tests.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class TestPodcastLibrary(unittest.TestCase):
    def setUp(self) -> None:
        # Import inside setUp so we can patch module globals per-test
        from vibevoice.config import config

        self._tmp = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.podcasts_dir = self.tmp_dir / "podcasts"
        self.podcasts_dir.mkdir(parents=True, exist_ok=True)

        # Patch config paths
        config.PODCASTS_DIR = self.podcasts_dir

        # Patch storage instance used by routes to point at temp file
        from vibevoice.models.podcast_storage import PodcastStorage
        from vibevoice.routes import podcasts as podcasts_routes

        self.storage = PodcastStorage(storage_file=self.podcasts_dir / "podcast_metadata.json")
        podcasts_routes.podcast_storage = self.storage

        # Import app after patching route dependencies
        from vibevoice.main import app
        from fastapi.testclient import TestClient

        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_search_download_delete(self) -> None:
        # Create dummy podcast files
        pid = "test-podcast-1"
        audio_path = self.podcasts_dir / f"{pid}.wav"
        script_path = self.podcasts_dir / f"{pid}.txt"
        audio_bytes = b"RIFF....WAVEfmt "  # not a valid wav, but sufficient for download plumbing
        audio_path.write_bytes(audio_bytes)
        script_path.write_text("Speaker 1: Hello")

        self.storage.add_podcast(
            podcast_id=pid,
            title="My Test Podcast",
            voices=["Alice", "Frank"],
            audio_path=audio_path,
            script_path=script_path,
            source_url="https://example.com/article",
            genre="News",
            duration="10 min",
        )

        # List
        r = self.client.get("/api/v1/podcasts")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["podcasts"][0]["id"], pid)

        # Search miss
        r = self.client.get("/api/v1/podcasts", params={"query": "does-not-match"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total"], 0)

        # Download
        r = self.client.get(f"/api/v1/podcasts/{pid}/download")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, audio_bytes)

        # Delete
        r = self.client.delete(f"/api/v1/podcasts/{pid}")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(audio_path.exists())
        self.assertFalse(script_path.exists())

        # List empty
        r = self.client.get("/api/v1/podcasts")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total"], 0)


if __name__ == "__main__":
    unittest.main()

