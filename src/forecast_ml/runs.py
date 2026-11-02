"""실험 실행마다 runs/<날짜시각-이름>/ 폴더를 만들고 설정을 남긴다."""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def _git_commit(repo: Path) -> str | None:
    """저장소의 현재 커밋. 커밋이 없거나 git이 없으면 None."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def new_run(runs_root: Path, name: str, config: dict, repo: Path) -> Path:
    """runs_root/<YYYYMMDD-HHMMSS-name>/를 만들고 config.json(설정·입력 파일 크기·커밋)을 쓴 뒤 경로를 돌려준다."""
    run_dir = runs_root / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs = {str(p): p.stat().st_size for p in map(Path, config.get("inputs", []))}
    record = {**config, "input_sizes": inputs, "git_commit": _git_commit(repo)}
    (run_dir / "config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run_dir


def write_json(path: Path, data: dict) -> None:
    """한글을 그대로 둔 JSON으로 쓴다."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
