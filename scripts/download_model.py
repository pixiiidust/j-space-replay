"""Download Qwen2.5-VL-7B-Instruct with a pinned revision + integrity check.

    uv run python scripts/download_model.py
    uv run python scripts/download_model.py --revision <commit-sha>   # reproducible pin
    uv run python scripts/download_model.py --check-only              # verify an existing cache

Qwen2.5-VL-7B-Instruct is Apache-2.0 and ungated — no HF token required. The
model is ~16 GB in fp16 safetensors; we load it 4-bit at runtime (bitsandbytes
NF4), so no separate quantized download is needed.

What this script guarantees:
  1. Revision pinning — downloads a specific commit (default "main"), and writes
     the *resolved* commit sha to scripts/model_revision.lock so a later run
     (or another machine) can reproduce the exact snapshot with --revision.
  2. File-list check — every file the HF repo advertises for that revision is
     present locally.
  3. Size check — each local file's byte size matches the repo metadata.
  4. Checksum check — for the large LFS files (the safetensors weight shards),
     the local sha256 is recomputed and compared to the sha256 HF stores in the
     file's LFS metadata. (HF only exposes sha256 for LFS-tracked files; small
     non-LFS files are covered by the size + presence checks and git's own
     content hashing.)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_REVISION = "main"  # override with a commit sha for a reproducible pin
LOCK_FILE = Path(__file__).with_name("model_revision.lock")

# Weight shards + configs we actually load. Video/tokenizer assets are pulled by
# the same snapshot; the allow-list keeps us from fetching unrelated extras.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "merges.txt", "*.py", "*.jinja"]


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_and_download(repo_id: str, revision: str, local_dir: Path | None) -> tuple[str, Path]:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    path = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(local_dir) if local_dir else None,
    )
    # Resolve the branch/tag we asked for to the concrete commit sha it pointed at.
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision)
    resolved = info.sha or revision
    _ = HF_HUB_CACHE  # (documented location; not needed further here)
    return resolved, Path(path)


def verify(repo_id: str, revision: str, local_path: Path) -> list[str]:
    """Return a list of problems (empty == verified)."""
    from huggingface_hub import HfApi

    problems: list[str] = []
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    checked_files = checked_hashes = 0

    for sib in info.siblings or []:
        # Only assets the snapshot allow-list would have fetched.
        if not any(Path(sib.rfilename).match(p) for p in ALLOW_PATTERNS):
            continue
        local = local_path / sib.rfilename
        if not local.exists():
            problems.append(f"missing file: {sib.rfilename}")
            continue
        checked_files += 1
        if sib.size is not None and local.stat().st_size != sib.size:
            problems.append(
                f"size mismatch {sib.rfilename}: local {local.stat().st_size} != repo {sib.size}"
            )
        lfs = getattr(sib, "lfs", None)
        want_sha = None
        if lfs is not None:
            want_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if want_sha:
            got = _sha256(local)
            checked_hashes += 1
            if got != want_sha:
                problems.append(f"sha256 mismatch {sib.rfilename}: {got} != {want_sha}")

    print(f"[verify] {checked_files} files checked for presence + size; "
          f"{checked_hashes} LFS shards checksummed.")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--revision", default=None,
                    help="commit sha / branch / tag (default: model_revision.lock, else 'main')")
    ap.add_argument("--local-dir", type=Path, default=None,
                    help="download into this dir instead of the shared HF cache")
    ap.add_argument("--check-only", action="store_true",
                    help="verify an already-downloaded snapshot; do not download")
    args = ap.parse_args(argv)

    # Revision precedence: explicit flag > lockfile > DEFAULT_REVISION.
    revision = args.revision
    if revision is None and LOCK_FILE.exists():
        revision = LOCK_FILE.read_text(encoding="utf-8").strip() or None
    revision = revision or DEFAULT_REVISION

    try:
        if args.check_only:
            from huggingface_hub import snapshot_download

            local = Path(snapshot_download(
                args.repo_id, revision=revision, allow_patterns=ALLOW_PATTERNS,
                local_files_only=True,
                local_dir=str(args.local_dir) if args.local_dir else None,
            ))
            resolved = revision
        else:
            print(f"[download] {args.repo_id} @ {revision}")
            resolved, local = resolve_and_download(args.repo_id, revision, args.local_dir)
            LOCK_FILE.write_text(resolved + "\n", encoding="utf-8")
            print(f"[download] resolved to commit {resolved} -> {LOCK_FILE.name}")
    except Exception as exc:  # noqa: BLE001 - surface a friendly message, not a traceback
        print(f"[error] download/resolve failed: {exc}", file=sys.stderr)
        return 2

    problems = verify(args.repo_id, resolved if not args.check_only else revision, local)
    if problems:
        print("[verify] FAILED:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    print(f"[verify] OK — snapshot at {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
