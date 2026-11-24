from pathlib import Path
from types import SimpleNamespace

import pytest

from reviews_pipeline.state import GhReleaseStore

REPO = "owner/private-data"


class FakeGh:
    """Models gh releases: assets per tag, and `--clobber` deleting before uploading."""

    def __init__(self, assets=None, releases=("state",), respond=None, fail_upload_of=None):
        self.assets = {"state": dict(assets or {})}
        for tag in releases:
            self.assets.setdefault(tag, {})
        self.releases = set(releases)
        self.respond = respond
        self.fail_upload_of = fail_upload_of
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if self.respond and (answer := self.respond(argv)) is not None:
            return SimpleNamespace(returncode=answer[0], stderr=answer[1], stdout="")
        verb, tag = tuple(argv[1:3]), argv[3] if len(argv) > 3 else None
        if verb == ("release", "view"):
            return self._result(tag in self.releases, "release not found")
        if verb == ("release", "create"):
            self.releases.add(tag)
            self.assets.setdefault(tag, {})
            return self._result(True)
        if tag not in self.releases:
            return self._result(False, "release not found")
        if verb == ("release", "download"):
            dest = Path(argv[argv.index("-D") + 1])
            wanted = [argv[i + 1] for i, a in enumerate(argv) if a == "--pattern"]
            found = [name for name in wanted if name in self.assets[tag]]
            for name in found:
                (dest / name).write_bytes(self.assets[tag][name])
            return self._result(bool(found), "no assets match the file pattern")
        if verb == ("release", "upload"):
            path = Path(argv[4])
            if "--clobber" in argv:
                self.assets[tag].pop(path.name, None)  # gh deletes first, then uploads
            if path.name == self.fail_upload_of:
                return self._result(False, "connection reset")
            self.assets[tag][path.name] = path.read_bytes()
            return self._result(True)
        raise AssertionError(f"unexpected gh call: {argv}")

    @staticmethod
    def _result(ok, stderr=""):
        return SimpleNamespace(returncode=0 if ok else 1, stderr="" if ok else stderr, stdout="")

    def verbs(self):
        return [tuple(argv[1:3]) for argv in self.calls]

    def uploaded_names(self):
        return [Path(a[4]).name for a in self.calls if a[1:3] == ["release", "upload"]]


def make_src(tmp_path, bronze_name="20261124T120000-head.parquet"):
    (tmp_path / "bronze").mkdir(parents=True)
    (tmp_path / "reviews.parquet").write_bytes(b"new-reviews")
    (tmp_path / "review_versions.parquet").write_bytes(b"new-versions")
    (tmp_path / "bronze" / bronze_name).write_bytes(b"raw")
    return tmp_path


def test_pull_downloads_release_assets(tmp_path):
    gh = FakeGh(assets={"reviews.parquet": b"r", "review_versions.parquet": b"v"})
    GhReleaseStore(REPO, runner=gh).pull(tmp_path)
    argv = gh.calls[0]
    assert argv[:9] == ["gh", "release", "download", "state", "-R", REPO, "-D", str(tmp_path),
                        "--clobber"]
    assert "reviews.parquet" in argv and "review_versions.parquet" in argv
    assert (tmp_path / "reviews.parquet").read_bytes() == b"r"


def test_pull_missing_release_is_not_an_error(tmp_path):
    GhReleaseStore(REPO, runner=FakeGh(releases=())).pull(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_pull_release_without_state_is_a_fresh_start(tmp_path):
    GhReleaseStore(REPO, runner=FakeGh()).pull(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_pull_other_errors_raise(tmp_path):
    gh = FakeGh(respond=lambda argv: (1, "HTTP 401: Bad credentials"))
    with pytest.raises(RuntimeError, match="401"):
        GhReleaseStore(REPO, runner=gh).pull(tmp_path)


def test_pull_restores_a_missing_state_file_from_prev(tmp_path):
    gh = FakeGh(assets={"review_versions.parquet": b"v", "reviews.parquet.prev": b"prev"})
    GhReleaseStore(REPO, runner=gh).pull(tmp_path)
    assert (tmp_path / "reviews.parquet").read_bytes() == b"prev"
    assert (tmp_path / "review_versions.parquet").read_bytes() == b"v"
    assert not (tmp_path / "reviews.parquet.prev").exists()


def test_pull_fails_loudly_when_state_is_incomplete_without_prev(tmp_path):
    gh = FakeGh(assets={"review_versions.parquet": b"v"})
    with pytest.raises(RuntimeError, match="incomplete"):
        GhReleaseStore(REPO, runner=gh).pull(tmp_path)


def test_pull_discards_stale_local_state(tmp_path):
    (tmp_path / "reviews.parquet").write_bytes(b"stale")
    GhReleaseStore(REPO, runner=FakeGh()).pull(tmp_path)
    assert not (tmp_path / "reviews.parquet").exists()


def test_push_creates_release_if_missing(tmp_path):
    gh = FakeGh(releases=())
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    verbs = gh.verbs()
    assert verbs.index(("release", "create")) < verbs.index(("release", "upload"))
    assert gh.assets["state"]["reviews.parquet"] == b"new-reviews"


def test_push_keeps_previous_state_as_prev_before_replacing(tmp_path):
    gh = FakeGh(assets={"reviews.parquet": b"old", "review_versions.parquet": b"oldv"})
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    names = gh.uploaded_names()
    assert names.index("reviews.parquet.prev") < names.index("reviews.parquet")
    assert gh.assets["state"]["reviews.parquet.prev"] == b"old"
    assert gh.assets["state"]["reviews.parquet"] == b"new-reviews"


def test_push_without_previous_asset_uploads_only_new(tmp_path):
    gh = FakeGh()
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    assert "reviews.parquet.prev" not in gh.assets["state"]
    assert gh.assets["state"]["reviews.parquet"] == b"new-reviews"


def test_interrupted_upload_is_recovered_by_the_next_pull(tmp_path):
    gh = FakeGh(
        assets={"reviews.parquet": b"old", "review_versions.parquet": b"oldv"},
        fail_upload_of="reviews.parquet",
    )
    store = GhReleaseStore(REPO, runner=gh)
    with pytest.raises(RuntimeError, match="connection reset"):
        store.push(make_src(tmp_path / "run1"))
    assert "reviews.parquet" not in gh.assets["state"]  # clobber deleted it, upload failed
    store.pull(tmp_path / "run2")
    assert (tmp_path / "run2" / "reviews.parquet").read_bytes() == b"old"
    assert (tmp_path / "run2" / "review_versions.parquet").read_bytes() == b"oldv"


def test_push_sends_bronze_to_a_monthly_release_without_clobber(tmp_path):
    gh = FakeGh()
    GhReleaseStore(REPO, runner=gh).push(make_src(tmp_path))
    assert "bronze-2026-11" in gh.releases
    assert gh.assets["bronze-2026-11"] == {"20261124T120000-head.parquet": b"raw"}
    assert "20261124T120000-head.parquet" not in gh.assets["state"]
    bronze_upload = next(a for a in gh.calls if a[1:3] == ["release", "upload"] and "bronze" in a[3])
    assert "--clobber" not in bronze_upload
