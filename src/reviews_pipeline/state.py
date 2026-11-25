"""State kept in the assets of a release in a private GitHub repository."""

import subprocess
import tempfile
from pathlib import Path

STATE_FILES = ("reviews.parquet", "review_versions.parquet")
# Written by the topic enrichment; absent before its first run, and never required.
OPTIONAL_STATE_FILES = ("review_topics.parquet", "topic_model.zip")
_ALL_FILES = STATE_FILES + OPTIONAL_STATE_FILES
_MISSING = ("release not found", "no assets match", "no assets to download")


class GhReleaseStore:
    """Pipeline state stored as assets of a release in a private GitHub repository."""

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

    def prepare(self) -> None:
        """Create the state release if missing, so a broken setup fails before collecting.

        Raises:
            RuntimeError: If the release cannot be created, for example in an empty repository.
        """
        self._ensure_release(self.tag)

    def pull(self, dest: Path) -> None:
        """Download the state files (and the optional enrichment files), restoring any missing one from its `.prev` backup.

        `gh release upload --clobber` deletes the old asset before uploading the new one, so
        an interrupted push can leave a state file missing while its `.prev` copy survives.

        Args:
            dest: Directory to download into, created if missing. Local copies of the
                state files are removed first, so stale state is never loaded.

        Raises:
            RuntimeError: If `gh` fails for any reason other than a missing release or asset,
                or if only part of the state exists and cannot be restored.
        """
        dest.mkdir(parents=True, exist_ok=True)
        for name in _ALL_FILES:
            (dest / name).unlink(missing_ok=True)
            (dest / f"{name}.prev").unlink(missing_ok=True)
        self._download(dest, _ALL_FILES + tuple(f"{n}.prev" for n in _ALL_FILES))
        present = [name for name in STATE_FILES if (dest / name).exists()]
        backups = [name for name in STATE_FILES if (dest / f"{name}.prev").exists()]
        if present or backups:
            for name in STATE_FILES:
                if not (dest / name).exists():
                    if name not in backups:
                        raise RuntimeError(f"state is incomplete: {name} is missing, no backup")
                    (dest / f"{name}.prev").rename(dest / name)
        for name in OPTIONAL_STATE_FILES:
            if not (dest / name).exists() and (dest / f"{name}.prev").exists():
                (dest / f"{name}.prev").rename(dest / name)
        for name in _ALL_FILES:
            (dest / f"{name}.prev").unlink(missing_ok=True)

    def _upload(self, path: Path, clobber: bool, tag: str | None = None) -> None:
        extra = ["--clobber"] if clobber else []
        self._gh("release", "upload", tag or self.tag, str(path), "-R", self.repo, *extra)

    def _ensure_release(self, tag: str) -> None:
        if not self._gh("release", "view", tag, "-R", self.repo, allow_missing=True):
            self._gh("release", "create", tag, "-R", self.repo, "--title", tag,
                     "--notes", "pipeline data")

    def push(self, src: Path) -> None:
        """Upload state files (keeping the previous copy as `.prev`) and new bronze files.

        Bronze files go to one release per month (`bronze-YYYY-MM`, taken from the file
        name), because a release holds at most 1,000 assets.

        Args:
            src: Directory with the state Parquet files at its root and bronze
                files under `bronze/`, named `YYYYMMDDTHHMMSS-<mode>.parquet`.

        Raises:
            RuntimeError: If any `gh` command fails.
        """
        self._ensure_release(self.tag)
        for name in _ALL_FILES:
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
            tag = f"bronze-{path.name[:4]}-{path.name[4:6]}"
            self._ensure_release(tag)
            self._upload(path, clobber=False, tag=tag)
