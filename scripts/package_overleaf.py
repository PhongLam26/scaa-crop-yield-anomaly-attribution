from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "paper" / "latex_source"
ZIP_PATH = ROOT / "paper" / "overleaf_zip" / "scaa_crop_yield_anomaly_attribution.zip"


def main() -> None:
    required = [
        LATEX / "main.tex",
        LATEX / "references.bib",
        LATEX / "figures",
        LATEX / "tables",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing Overleaf inputs: {missing}")
    figures = sorted((LATEX / "figures").glob("*.png"))
    tables = sorted((LATEX / "tables").glob("*.tex"))
    if not figures:
        raise AssertionError("No figures found for Overleaf package")
    if not tables:
        raise AssertionError("No tables found for Overleaf package")

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in [LATEX / "main.tex", LATEX / "references.bib"]:
            z.write(path, path.relative_to(LATEX).as_posix())
        for path in figures + tables:
            z.write(path, path.relative_to(LATEX).as_posix())
        z.writestr(
            "README_OVERLEAF.txt",
            "Upload this zip to Overleaf. Compile main.tex. "
            "Reference metadata should be verified before formal journal submission.\n",
        )

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = set(z.namelist())
        for needed in ["main.tex", "references.bib", "README_OVERLEAF.txt"]:
            if needed not in names:
                raise AssertionError(f"Missing {needed} in zip")
    print(f"Overleaf package written: {ZIP_PATH}")


if __name__ == "__main__":
    main()
