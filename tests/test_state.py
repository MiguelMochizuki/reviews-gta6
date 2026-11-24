from pathlib import Path
from types import SimpleNamespace

import pytest

from reviews_pipeline.state import GhReleaseStore

REPO = "owner/private-data"


class FakeGh:
    """Records gh calls; `respond(argv)` may return (returncode, stderr)."""

    def __init__(self, respond=None, existing_assets=()):
        self.respond = respond or (lambda argv: (0, ""))
        self.existing = set(existing_assets)
        self.calls = []  # (argv, {path: bytes} for uploaded files)

    def __call__(self, argv, **kwargs):
        uploads = {}
        if argv[:3] == ["gh", "release", "upload"]:
            for arg in argv[4:]:
                if Path(arg).is_file():
                    uploads[arg] = Path(arg).read_bytes()
        self.calls.append((argv, uploads))
        code, stderr = self.respond(argv)
        if argv[:3] == ["gh", "release", "download"] and code == 0:
            dest = Path(argv[argv.index("-D") + 1])
            for name in (argv[i + 1] for i, a in enumerate(argv) if a == "--pattern"):
                if name in self.existing:
                    (dest / name).write_bytes(b"old")
        return SimpleNamespace(returncode=code, stderr=stderr, stdout="")

    def verbs(self):
        return [tuple(argv[1:3]) for argv, _ in self.calls]


def test_pull_downloads_release_assets(tmp_path):
    gh = FakeGh()
    GhReleaseStore(REPO, runner=gh).pull(tmp_path)
    argv = gh.calls[0][0]
    assert argv[:9] == ["gh", "release", "download", "state", "-R", REPO, "-D", str(tmp_path),
                        "--clobber"]
    assert "reviews.parquet" in argv and "review_versions.parquet" in argv


def test_pull_missing_release_is_not_an_error(tmp_path):
    gh = FakeGh(lambda argv: (1, "release not found"))
    GhReleaseStore(REPO, runner=gh).pull(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_pull_other_errors_raise(tmp_path):
    gh = FakeGh(lambda argv: (1, "HTTP 401: Bad credentials"))
    with pytest.raises(RuntimeError, match="401"):
        GhReleaseStore(REPO, runner=gh).pull(tmp_path)


def make_src(tmp_path):
    (tmp_path / "bronze").mkdir()
    (tmp_path / "reviews.parquet").write_bytes(b"new-reviews")
    (tmp_path / "bronze" / "run-1.parquet").write_bytes(b"raw")
    return tmp_path


def test_push_creates_release_if_missing(tmp_path):
    gh = FakeGh(lambda argv: (1, "release not found") if argv[1:3] == ["release", "view"] else (0, ""))
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    verbs = gh.verbs()
    assert verbs.index(("release", "create")) < verbs.index(("release", "upload"))


def test_push_keeps_previous_state_as_prev(tmp_path):
    gh = FakeGh(existing_assets={"reviews.parquet"})
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    uploads = [(argv, up) for argv, up in gh.calls if argv[1:3] == ["release", "upload"]]
    state_uploads = [(a, u) for a, u in uploads if any("reviews.parquet" in p for p in u)]
    first_argv, first_files = state_uploads[0]
    second_argv, second_files = state_uploads[1]
    assert next(iter(first_files)).endswith("reviews.parquet.prev")
    assert list(first_files.values()) == [b"old"]
    assert next(iter(second_files)) == str(tmp_path / "reviews.parquet")
    assert list(second_files.values()) == [b"new-reviews"]
    assert "--clobber" in first_argv and "--clobber" in second_argv


def test_push_without_previous_asset_uploads_only_new(tmp_path):
    def respond(argv):
        if argv[1:3] == ["release", "download"]:
            return 1, "no assets match the file pattern"
        return 0, ""

    gh = FakeGh(respond)
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    state_uploads = [u for a, u in gh.calls if a[1:3] == ["release", "upload"]
                     and any("reviews.parquet" in p for p in u)]
    assert len(state_uploads) == 1
    assert not next(iter(state_uploads[0])).endswith(".prev")


def test_push_uploads_bronze_files_without_clobber(tmp_path):
    gh = FakeGh()
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    bronze = [a for a, u in gh.calls if any("run-1.parquet" in p for p in u)]
    assert len(bronze) == 1
    assert "--clobber" not in bronze[0]
