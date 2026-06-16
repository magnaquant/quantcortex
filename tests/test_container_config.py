from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_IMPORTS = [
    "import portfolio",
    "import data",
    "import alpha",
    "import backtest",
    "import execution",
    "import strategies",
]


def test_container_commands_use_quantcortex_namespace():
    for name in ("Dockerfile", "docker-compose.yml"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "import quantcortex." in text
        assert not any(legacy in text for legacy in LEGACY_IMPORTS)


def test_docker_context_excludes_local_data_and_secrets():
    entries = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert {".git", ".venv", ".env", "local_data", "reports"} <= entries
