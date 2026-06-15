from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "paper" / "latex_source"
ZIP_PATH = ROOT / "paper" / "overleaf_zip" / "scaa_crop_yield_anomaly_attribution.zip"
MANIFEST = ROOT / "paper" / "DATA_MANIFEST.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest_zip_checksum() -> None:
    if not MANIFEST.exists():
        return
    rel = r"paper\overleaf_zip\scaa_crop_yield_anomaly_attribution.zip"
    size = ZIP_PATH.stat().st_size
    digest = sha256_file(ZIP_PATH)
    replacement = f"| `{rel}` | {size} | `{digest}` |"
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    updated = [replacement if line.startswith(f"| `{rel}` |") else line for line in lines]
    MANIFEST.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> None:
    required = [
        LATEX / "main.tex",
        LATEX / "supplement.tex",
        LATEX / "references.bib",
        LATEX / "figures",
        LATEX / "tables",
        LATEX / "supplement",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing Overleaf inputs: {missing}")
    figures = sorted((LATEX / "figures").glob("*.png"))
    tables = sorted((LATEX / "tables").glob("*.tex"))
    supplement = sorted((LATEX / "supplement").glob("*.tex"))
    if not figures:
        raise AssertionError("No figures found for Overleaf package")
    if not tables:
        raise AssertionError("No tables found for Overleaf package")
    if not supplement:
        raise AssertionError("No supplement tables found for Overleaf package")

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in [LATEX / "main.tex", LATEX / "supplement.tex", LATEX / "references.bib"]:
            z.write(path, path.relative_to(LATEX).as_posix())
        for path in figures + tables + supplement:
            z.write(path, path.relative_to(LATEX).as_posix())
        z.writestr(
            "README_OVERLEAF.txt",
            "Upload this zip to Overleaf. Compile main.tex for the manuscript. "
            "Compile supplement.tex separately if the venue allows supplementary material. "
            "Reference metadata should be verified before formal journal submission.\n",
        )

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = set(z.namelist())
        for needed in ["main.tex", "supplement.tex", "references.bib", "README_OVERLEAF.txt"]:
            if needed not in names:
                raise AssertionError(f"Missing {needed} in zip")
        csv_files = [name for name in names if name.lower().endswith(".csv")]
        if csv_files:
            raise AssertionError(f"CSV files should not be included in Overleaf zip: {csv_files}")
    update_manifest_zip_checksum()
    print(f"Overleaf package written: {ZIP_PATH}")


if __name__ == "__main__":
    main()
