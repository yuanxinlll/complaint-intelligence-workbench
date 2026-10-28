"""Build a reviewable code-only collaboration snapshot without any upload."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "exports/share-ready"
    archive = ROOT / "exports/complaint-intelligence-workbench-code.zip"
    if output.exists() or archive.exists():
        raise FileExistsError("Share snapshot already exists; preserve it and review before rebuilding.")
    files = [ROOT / name for name in [".gitignore", "README.md", "STATUS.md", "requirements.txt", "requirements-evaluation.txt"]]
    files += sorted((ROOT / "src").glob("*.py"))
    files += sorted((ROOT / "tests").glob("test_*.py"))
    files += [ROOT / "reports" / name for name in ["SHARED_RESULTS.md", "test_confusion_matrix.png", "validation_review_curve.png"]]
    manifest = []
    for source in files:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular source file: {source.name}")
        relative = source.relative_to(ROOT)
        if source.suffix not in {".py", ".md", ".txt", ".png"} and source.name != ".gitignore":
            raise ValueError(f"Unexpected share file: {relative}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest.append({"path": str(relative), "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in manifest:
            bundle.write(output / item["path"], f"complaint-intelligence-workbench/{item['path']}")
    (ROOT / "exports/share_manifest.json").write_text(json.dumps({"files": manifest, "raw_data_included": False, "models_included": False, "uploaded": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "zip_bytes": archive.stat().st_size, "snapshot": str(output), "archive": str(archive), "uploaded": False}, indent=2))


if __name__ == "__main__":
    main()
