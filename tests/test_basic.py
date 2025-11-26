from pathlib import Path


def test_repository_layout_includes_src_and_tests():
    """Smoke-test that the repository layout expected by other tests is present."""

    project_root = Path(__file__).resolve().parents[1]
    assert (project_root / "src").is_dir()
    assert (project_root / "tests").is_dir()
