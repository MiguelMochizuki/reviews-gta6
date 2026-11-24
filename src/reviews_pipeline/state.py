"""State kept in the assets of a release in a private GitHub repository."""

import subprocess
import tempfile
from pathlib import Path

STATE_FILES = ("reviews.parquet", "review_versions.parquet")
_MISSING = ("release not found", "no assets match")


class GhReleaseStore:
    def __init__(self, repo: str, tag: str = "state", runner=subprocess.run):
        self.repo = repo
        self.tag = tag
        self._runner = runner

    def _gh(self, *args: str, allow_missing: bool = False) -> bool:
        """Run gh; return False when the release or asset is missing and that is allowed."""
        result = self._runner(["gh", *args], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        if allow_missing and any(phrase in result.stderr for phrase in _MISSING):
            return False
        raise RuntimeError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()[:200]}")

    def _download(self, dest: Path, names=STATE_FILES) -> bool:
        patterns = [arg for name in names for arg in ("--pattern", name)]
        return self._gh(
            "release", "download", self.tag, "-R", self.repo, "-D", str(dest), "--clobber",
            *patterns, allow_missing=True,
        )

    def pull(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        self._download(dest)

    def _upload(self, path: Path, clobber: bool) -> None:
        extra = ["--clobber"] if clobber else []
        self._gh("release", "upload", self.tag, str(path), "-R", self.repo, *extra)

    def push(self, src: Path) -> None:
        if not self._gh("release", "view", self.tag, "-R", self.repo, allow_missing=True):
            self._gh("release", "create", self.tag, "-R", self.repo,
                     "--title", self.tag, "--notes", "pipeline state")
        for name in STATE_FILES:
            current = src / name
            if not current.exists():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                if self._download(Path(tmp), (name,)) and (Path(tmp) / name).exists():
                    previous = Path(tmp) / f"{name}.prev"
                    (Path(tmp) / name).rename(previous)
                    self._upload(previous, clobber=True)
            self._upload(current, clobber=True)
        for path in sorted((src / "bronze").glob("*.parquet")):
            self._upload(path, clobber=False)
