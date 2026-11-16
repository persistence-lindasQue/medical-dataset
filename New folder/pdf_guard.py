#!/usr/bin/env python3
"""pdf_guard: flag PDFs that contain hidden or suspicious text before ingestion.

Checks (per page):
  - white / near-white text (assumes a white page background)
  - tiny text (font size below --min-size points)
  - text positioned outside the visible page area
  - invisible text render mode (3 Tr) in page content streams
  - instruction-like phrases aimed at AI models, in visible or hidden text
  - the same phrases in document metadata and annotations

Flagged files are moved to a quarantine folder; clean files stay put.
Exit code: 0 = all clean, 1 = at least one file flagged, 2 = error.

Limits: this is a heuristic filter, not a guarantee. It assumes a white page
background, cannot see text hidden behind images or shapes, and the phrase
list is short. Treat a clean result as "no known tricks found".
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

# Phrases aimed at manipulating a model. Case-insensitive.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions?|prompts?|rules?)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier)",
    r"(you are|act as|pretend to be) (now )?(a |an )?(different|new|unrestricted|jailbroken)",
    r"system prompt",
    r"do not (tell|reveal|mention) (this|the user)",
    r"as an? (ai|language model|assistant)[, ]+(you )?(must|should|will)",
    r"(always|must) (recommend|say|answer|respond) (that )?",
    r"override (your )?(safety|guidelines|instructions)",
    r"new instructions?:",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I)

TR3_RE = re.compile(rb"(?<![\w.])3\s+Tr(?![\w])")


def _is_near_white(color, threshold: float = 0.95) -> bool:
    """True if a PDF fill colour is white or nearly white."""
    if color is None:
        return False
    if isinstance(color, (int, float)):
        return float(color) >= threshold  # grayscale as a bare number
    try:
        vals = [float(v) for v in color]
    except (TypeError, ValueError):
        return False  # pattern / named colour: cannot judge
    if len(vals) == 1:  # grayscale
        return vals[0] >= threshold
    if len(vals) == 3:  # RGB
        return all(v >= threshold for v in vals)
    if len(vals) == 4:  # CMYK: white means all inks near zero
        return all(v <= 1 - threshold for v in vals)
    return False


def _snippet(chars: list[dict], limit: int = 120) -> str:
    text = "".join(c.get("text", "") for c in chars).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit] + ("..." if len(text) > limit else "")


def _dark_rects(page) -> list[tuple[float, float, float, float]]:
    """Boxes filled with a non-white colour (table headers, banners, callouts)."""
    boxes = []
    for r in page.rects:
        if r.get("fill") and not _is_near_white(r.get("non_stroking_color")):
            boxes.append((float(r["x0"]), float(r["top"]), float(r["x1"]), float(r["bottom"])))
    return boxes


def _on_dark_box(ch: dict, boxes: list[tuple[float, float, float, float]]) -> bool:
    cx = (float(ch["x0"]) + float(ch["x1"])) / 2
    cy = (float(ch["top"]) + float(ch["bottom"])) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def scan_page(page, page_no: int, min_size: float, min_hidden: int = 30) -> list[dict]:
    findings: list[dict] = []
    width, height = float(page.width), float(page.height)
    margin = 2.0
    boxes = _dark_rects(page)

    white, tiny, offpage = [], [], []
    for ch in page.chars:
        if not ch.get("text", "").strip():
            continue
        # White text on a coloured box (table header, banner) is normal in papers.
        if _is_near_white(ch.get("non_stroking_color")) and not _on_dark_box(ch, boxes):
            white.append(ch)
        if float(ch.get("size", 99)) < min_size:
            tiny.append(ch)
        x0, x1 = float(ch["x0"]), float(ch["x1"])
        top, bottom = float(ch["top"]), float(ch["bottom"])
        if x1 < -margin or x0 > width + margin or bottom < -margin or top > height + margin:
            offpage.append(ch)

    for kind, chars, why in (
        ("white_text", white, "text coloured white or near-white"),
        ("tiny_text", tiny, f"text smaller than {min_size}pt"),
        ("offpage_text", offpage, "text placed outside the visible page"),
    ):
        # A few stray characters are normal (superscripts, figure labels, logos).
        # Hidden instructions are sentence-length, so require a minimum amount.
        if len(chars) >= min_hidden:
            findings.append(
                {"page": page_no, "type": kind, "detail": why,
                 "char_count": len(chars), "sample": _snippet(chars)}
            )

    visible_text = page.extract_text() or ""
    for m in INJECTION_RE.finditer(visible_text):
        findings.append(
            {"page": page_no, "type": "injection_phrase",
             "detail": "instruction-like phrase in page text", "sample": m.group(0)}
        )
    return findings


def scan_render_mode(reader: PdfReader) -> list[dict]:
    """Detect invisible text render mode in page content streams."""
    findings = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            contents = page.get_contents()
            data = contents.get_data() if contents is not None else b""
        except Exception:
            continue
        if TR3_RE.search(data):
            findings.append(
                {"page": i, "type": "invisible_render_mode",
                 "detail": "content stream sets text render mode 3 (invisible)"}
            )
    return findings


def scan_metadata_and_annots(reader: PdfReader) -> list[dict]:
    findings = []
    meta = reader.metadata or {}
    for key, val in meta.items():
        if isinstance(val, str) and INJECTION_RE.search(val):
            findings.append(
                {"page": 0, "type": "injection_phrase_metadata",
                 "detail": f"instruction-like phrase in metadata field {key}",
                 "sample": val[:120]}
            )
    for i, page in enumerate(reader.pages, start=1):
        for annot_ref in page.get("/Annots") or []:
            try:
                annot = annot_ref.get_object()
                contents = str(annot.get("/Contents", "") or "")
            except Exception:
                continue
            if contents and INJECTION_RE.search(contents):
                findings.append(
                    {"page": i, "type": "injection_phrase_annotation",
                     "detail": "instruction-like phrase in an annotation",
                     "sample": contents[:120]}
                )
    return findings


def scan_pdf(path: Path, min_size: float, min_hidden: int = 30) -> dict:
    result = {"file": str(path), "status": "clean", "findings": [], "error": None}
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            result.update(status="error", error="encrypted PDF; not scanned")
            return result
        findings = scan_render_mode(reader) + scan_metadata_and_annots(reader)
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                findings.extend(scan_page(page, i, min_size, min_hidden))
        result["findings"] = findings
        if findings:
            result["status"] = "flagged"
    except Exception as exc:  # unreadable or malformed file
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", type=Path, help="a PDF file or a folder of PDFs")
    ap.add_argument("--quarantine", type=Path, default=Path("quarantine"),
                    help="where flagged/unreadable PDFs are moved")
    ap.add_argument("--report", type=Path, default=Path("guard_report.json"))
    ap.add_argument("--min-size", type=float, default=4.0,
                    help="font sizes below this (pt) count as tiny text")
    ap.add_argument("--min-hidden-chars", type=int, default=30,
                    help="hidden/tiny/off-page chars needed on a page before flagging (default 30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; do not move any files")
    args = ap.parse_args(argv)

    if args.input.is_dir():
        pdfs = sorted(args.input.rglob("*.pdf"))
    elif args.input.is_file():
        pdfs = [args.input]
    else:
        print(f"not found: {args.input}", file=sys.stderr)
        return 2

    results = [scan_pdf(p, args.min_size, args.min_hidden_chars) for p in pdfs]

    if not args.dry_run:
        args.quarantine.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r["status"] != "clean":
                src = Path(r["file"])
                dest = args.quarantine / src.name
                n = 1
                while dest.exists():
                    dest = args.quarantine / f"{src.stem}_{n}{src.suffix}"
                    n += 1
                shutil.move(str(src), str(dest))
                r["moved_to"] = str(dest)

    args.report.write_text(json.dumps(results, indent=2))

    counts = {s: sum(r["status"] == s for r in results) for s in ("clean", "flagged", "error")}
    print(f"scanned {len(results)} PDF(s): {counts['clean']} clean, "
          f"{counts['flagged']} flagged, {counts['error']} unreadable")
    for r in results:
        if r["status"] == "flagged":
            kinds = sorted({f["type"] for f in r["findings"]})
            print(f"  FLAGGED {Path(r['file']).name}: {', '.join(kinds)}")
        elif r["status"] == "error":
            print(f"  ERROR   {Path(r['file']).name}: {r['error']}")
    print(f"report: {args.report}")
    return 0 if counts["flagged"] == 0 and counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())