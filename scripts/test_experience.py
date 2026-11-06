"""CPU checks for experience packages: sealing, selection, episode copies, readers."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.packages import (
    PACKAGE_TYPE, ExperienceCatalog, ExperienceError, build_package, load_package,
    materialize, release_workspace, write_catalog,
)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        src = root / "src"
        (src / "skills").mkdir(parents=True)
        (src / "README.md").write_text("stack bowls: approach from above\n" * 50)
        (src / "skills" / "grasp.md").write_text("close gripper after contact")
        (src / "frame.png").write_bytes(b"\x89PNG" + b"0" * 64)
        self.packages = root / "packages"
        self.pkg = build_package(src, self.packages / "stack_bowls", "stack_bowls",
                                 metadata=dict(source="unit-test"))
        self.catalog = write_catalog(
            root / "catalog.json", "exp-r1", "packages", [self.pkg],
            {"stack_bowls": ["README.md"]},
            {"stack_bowls": dict(bundle="stack_bowls", kind="original_preparation",
                                 base_task="stack_bowls", reason="own demonstrations"),
             "stack_bowls_random": dict(bundle="stack_bowls", kind="base_task_transfer",
                                        base_task="stack_bowls", reason="same procedure")})
        self.episodes = root / "episodes"

    def tearDown(self):
        for directory, _, _ in os.walk(self.tmp.name):
            os.chmod(directory, 0o755)
        self.tmp.cleanup()

    def test_manifest_is_sensenova_compatible(self):
        manifest = json.loads((self.pkg.root / "manifest.json").read_text())
        self.assertEqual(manifest["package_type"], PACKAGE_TYPE)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(set(manifest["files"]), {"README.md", "skills/grasp.md", "frame.png"})
        with self.assertRaises(ExperienceError):
            build_package(self.packages / "stack_bowls" / "payload",
                          self.packages / "stack_bowls", "stack_bowls")

    def test_tampering_is_detected(self):
        (self.pkg.payload / "skills" / "grasp.md").write_text("open gripper")
        with self.assertRaises(ExperienceError):
            load_package(self.pkg.root)

    def test_extra_file_and_symlink_rejected(self):
        (self.pkg.payload / "extra.txt").write_text("x")
        with self.assertRaises(ExperienceError):
            load_package(self.pkg.root)
        (self.pkg.payload / "extra.txt").unlink()
        os.symlink("/etc/hostname", self.pkg.payload / "link.txt")
        with self.assertRaises(ExperienceError):
            load_package(self.pkg.root)

    def test_explicit_mapping_no_fallback(self):
        self.assertEqual(self.catalog.select("stack_bowls_random").kind, "base_task_transfer")
        with self.assertRaises(ExperienceError):
            self.catalog.select("unknown_task")

    def test_inconsistent_role_rejected(self):
        doc = json.loads(Path(self.tmp.name, "catalog.json").read_text())
        doc["tasks"]["stack_bowls_random"]["kind"] = "original_preparation"
        bad = Path(self.tmp.name, "bad.json")
        bad.write_text(json.dumps(doc))
        with self.assertRaises(ExperienceError):
            ExperienceCatalog.load(bad)

    def test_catalog_pins_manifest(self):
        manifest = self.pkg.root / "manifest.json"
        doc = json.loads(manifest.read_text())
        doc["metadata"]["note"] = "changed"
        manifest.write_text(json.dumps(doc))
        with self.assertRaises(ExperienceError):
            materialize(self.catalog, "stack_bowls", self.episodes / "e1")

    def test_episode_copy_readers_and_access_log(self):
        events = []
        ws = materialize(self.catalog, "stack_bowls", self.episodes / "e1", on_access=events.append)
        other = materialize(self.catalog, "stack_bowls", self.episodes / "e2")
        self.assertNotEqual(ws.root, other.root)
        self.assertIn("skills/grasp.md", ws.list())
        self.assertEqual(len(ws.read_text("README.md", limit=10)), 10)
        self.assertLessEqual(len(ws.read_text("README.md", limit=10 ** 9)), ws.max_text_chars)
        data, mime = ws.read_image("frame.png")
        self.assertEqual(mime, "image/png")
        for bad in ("../catalog.json", "/etc/hostname", "skills/../README.md"):
            with self.assertRaises(ExperienceError):
                ws.read_text(bad)
        with self.assertRaises(ExperienceError):
            ws.read_image("README.md")
        self.assertFalse(os.stat(ws.root / "README.md").st_mode & 0o222)  # read-only copy
        if os.geteuid() != 0:   # root ignores file modes
            with self.assertRaises(PermissionError):
                (ws.root / "README.md").write_text("edited")
        self.assertEqual([e["op"] for e in events], ["list", "read_text", "read_text", "read_image"])
        self.assertIn("stack_bowls", ws.instructions())
        self.assertEqual((self.pkg.payload / "README.md").read_text()[:5], "stack")
        release_workspace(ws)
        self.assertFalse(ws.root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
