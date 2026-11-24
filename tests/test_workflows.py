import re
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "collect.yml"


def text():
    return WORKFLOW.read_text()


def test_collect_workflow_has_schedule_and_concurrency():
    t = text()
    assert 'cron: "7 * * * *"' in t
    assert 'cron: "37 3 * * *"' in t
    assert "concurrency:" in t
    assert "group: collect" in t
    assert "cancel-in-progress: false" in t


def test_collect_workflow_has_no_pull_request_trigger():
    assert "pull_request" not in text()


def test_collect_workflow_skips_forks():
    assert "github.event.repository.fork == false" in text()


def test_collect_workflow_passes_secrets_only_as_env():
    t = text()
    assert sorted(set(re.findall(r"secrets\.(\w+)", t))) == ["AUTHOR_SALT", "STATE_TOKEN"]
    assert "${{ secrets.STATE_TOKEN }}" in t
    assert "${{ secrets.AUTHOR_SALT }}" in t
    assert "set -x" not in t
    assert not re.search(r"echo[^\n]*secrets\.", t)


def test_collect_workflow_reports_failure_in_one_issue():
    t = text()
    assert "if: failure()" in t
    assert "Collection run failed" in t
    assert "issues: write" in t


def test_ci_runs_tools_without_resyncing_default_groups():
    ci = (WORKFLOW.parent / "ci.yml").read_text()
    for line in ci.splitlines():
        if "uv run" in line:
            assert "--no-sync" in line, line  # a bare `uv run` would install the notebook group


def test_python_version_is_pinned():
    assert (WORKFLOW.parents[2] / ".python-version").read_text().strip() == "3.12"


def pages_job():
    t = text()
    return t[t.index("\n  deploy:"):]


def test_collect_builds_gold_only_after_a_successful_collection():
    t = text()
    assert t.index("name: Collect\n") < t.index("name: Build site data")
    step = t[t.index("name: Build site data"):t.index("upload-pages-artifact")]
    assert "if:" not in step  # the default is success(), so a failed collection never publishes
    assert "--state data/state" in step and "--out site/data" in step


def test_pages_job_has_no_secrets_and_minimal_permissions():
    job = pages_job()
    assert "secrets." not in job
    assert "needs: collect" in job
    assert "environment:" in job and "name: github-pages" in job
    assert "pages: write" in job and "id-token: write" in job
    assert "contents: write" not in job and "issues: write" not in job


def test_pages_artifact_contains_only_the_site_directory():
    t = text()
    assert t.count("upload-pages-artifact") == 1
    step = t[t.index("upload-pages-artifact"):]
    assert "path: site" in step.split("\n\n")[0] or "path: site\n" in step
    assert "path: data" not in t and "upload-artifact@" not in t
