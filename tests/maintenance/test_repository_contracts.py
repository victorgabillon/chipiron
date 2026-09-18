"""Safety contracts for validation and release publication workflows."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pr_validation_has_no_publication_credentials_or_push() -> None:
    """PR and canonical-branch validation cannot authenticate or publish images."""
    text = (ROOT / ".github/workflows/ci.yaml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert "pull_request" in workflow["on"]
    assert "main" in workflow["on"]["push"]["branches"]
    assert "maintenance/chipiron-canonicalization" in workflow["on"]["push"]["branches"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets." not in text
    assert "docker/login-action" not in text
    assert "docker push" not in text
    assert "push: true" not in text
    assert "python -m tox" in text


def test_publication_requires_tag_quality_and_container_smoke() -> None:
    """Only a version-tag release may publish an image after all validation gates."""
    workflow = yaml.load(
        (ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"push"}
    assert workflow["on"]["push"] == {"tags": ["v*"]}
    job = workflow["jobs"]["docker"]
    assert job["needs"] == ["quality"]
    assert "refs/tags/v" in job["if"]
    assert "victorgabillon/chipiron" in job["if"]
    steps = job["steps"]
    build = next(
        i
        for i, step in enumerate(steps)
        if "docker/build-push-action" in step.get("uses", "")
    )
    smoke = next(
        i for i, step in enumerate(steps) if "docker run" in step.get("run", "")
    )
    login = next(
        i
        for i, step in enumerate(steps)
        if "docker/login-action" in step.get("uses", "")
    )
    push = next(
        i for i, step in enumerate(steps) if "docker push" in step.get("run", "")
    )
    assert build < smoke < login < push
    assert steps[build]["with"]["push"] == "false"
    assert steps[build]["with"]["load"] == "true"
    assert workflow["jobs"]["publish"]["needs"] == ["quality"]
