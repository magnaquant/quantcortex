from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILES = tuple(sorted((REPO_ROOT / "requirements").glob("*.lock")))
SUPPORTED_PYTHONS = ("3.11", "3.12", "3.13", "3.14")


@pytest.mark.parametrize("lock_path", LOCK_FILES, ids=lambda path: path.name)
@pytest.mark.parametrize("python_version", SUPPORTED_PYTHONS)
def test_exported_locks_have_one_active_pin_per_package_on_ci(
    lock_path: Path,
    python_version: str,
):
    environment = default_environment()
    environment.update(
        {
            "extra": "",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "python_full_version": f"{python_version}.0",
            "python_version": python_version,
            "sys_platform": "linux",
        }
    )

    active: dict[str, list[str]] = defaultdict(list)
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            active[canonicalize_name(requirement.name)].append(
                str(requirement.specifier)
            )

    conflicts = {
        name: sorted(set(specifiers))
        for name, specifiers in active.items()
        if len(set(specifiers)) > 1
    }
    assert not conflicts, (
        f"{lock_path.name} activates conflicting pins on Python "
        f"{python_version}: {conflicts}"
    )
