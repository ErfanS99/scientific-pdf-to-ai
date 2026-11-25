#!/usr/bin/env python3
"""Docling PDF extraction with page-range checkpoints for Google Colab.

This variant automatically falls back to PyPdfium2 after a worker SIGTRAP.
It retains the original V3.1 checkpoint profile for resumption.


- Keeps original source PDF unchanged; uses Docling's inclusive 1-based page_range.
- Reuse one Docling converter across a bounded group of page batches, then recycle the process.
- Writes checkpoints to Google Drive with per-file SHA-256 integrity manifests.
- Resumes without reconverting verified batches, with exact source + config checks.
- Retains page-cited Markdown, Docling structure, figures, tables, and RAG chunks.
- OCR is off by default for born-digital PDFs with selectable text.

Scientific limitation: figures and equations still need source verification; objects
crossing a batch boundary may be reconstructed as separate pieces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

VERSION = "3.0"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MARKER = "_COMPLETE.json"


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def sha_file(path: Path, buf=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(buf):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def docling_version() -> str:
    from importlib.metadata import version
    return version("docling")


def make_profile(pdf: Path, source_sha: str, args) -> dict:
    return {
        "pipeline_version": VERSION,
        "source_filename": pdf.name,
        "source_bytes": pdf.stat().st_size,
        "source_sha256": source_sha,
        "docling_version": docling_version(),
        "batch_pages": args.batch_pages,
        "ocr": "off",
        "formula": args.formula,
        "image_scale": args.image_scale,
        "max_tokens": args.max_tokens,
        "device": args.device,
        "threads": args.threads,
        "tokenizer": TOKENIZER_NAME,
        "table_mode": "accurate",
    }


def file_manifest(folder: Path) -> dict:
    result = {}
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.name != MARKER:
            rel = file.relative_to(folder).as_posix()
            result[rel] = {"bytes": file.stat().st_size, "sha256": sha_file(file)}
    return result


def validate_checkpoint(folder: Path, profile_id: str, start: int, end: int) -> tuple[bool, str]:
    marker = folder / MARKER
    if not marker.is_file():
        return False, "missing completion marker"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("profile_id") != profile_id or data.get("start") != start or data.get("end") != end:
            return False, "source/options/page range mismatch"
        files = data.get("files", {})
        expected = {"batch.md", "rag_chunks.jsonl", "structure.json", "tables_index.jsonl", "figures_to_review.jsonl"}
        if not expected.issubset(files):
            return False, "missing mandatory checkpoint files"
        md_pages = [f for f in files if f.startswith("pages/") and f.endswith(".md")]
        if len(md_pages) != (end - start + 1):
            return False, "unexpected per-page Markdown file count"
        for rel, info in files.items():
            f = folder / rel
            if not f.is_file() or f.stat().st_size != info["bytes"]:
                return False, f"missing/incomplete {rel}"
            if sha_file(f) != info["sha256"]:
                return False, f"checksum mismatch {rel}"
        return True, "verified"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"damaged marker: {exc}"


def page_numbers(item) -> list[int]:
    return sorted({int(p.page_no) for p in (getattr(item, "prov", None) or [])})


def normalize_pages(doc, start: int, end: int):
    """Accept absolute source page ids, or correct a clearly local 1..N range."""
    keys = sorted(int(k) for k in doc.pages)
    expected = list(range(start, end + 1))
    if keys == expected:
        return "absolute"
    if start > 1 and keys == list(range(1, len(expected) + 1)):
        offset = start - 1
        remapped = {}
        for old, pg in doc.pages.items():
            pg.page_no = int(old) + offset
            remapped[int(old) + offset] = pg
        doc.pages = remapped
        for item, _level in doc.iterate_items():
            for prov in (getattr(item, "prov", None) or []):
                prov.page_no += offset
        return "rebased_local"
    raise RuntimeError(f"Docling returned page ids {keys[:5]}... expected {expected[0]}..{expected[-1]}")


def load_tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    t.model_max_length = 10**9
    return t


def chunk_by_tokens_exact(text: str, tokenizer, max_tokens: int):
    """Slice text on token boundaries without dropping words or characters."""
    if not text:
        return []
    # Character offsets from the fast tokenizer; prefer whitespace before cap.
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for lossless token offsets")
    segments = []
    index = 0
    while index < len(text):
        remainder = text[index:]
        enc = tokenizer(remainder, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
        offsets = enc["offset_mapping"]
        if len(offsets) <= max_tokens:
            segments.append(remainder)
            break
        last = offsets[max_tokens - 1][1]
        if last <= 0:
            raise RuntimeError("Tokenizer gave nonprogressing character offsets")
        candidates = [remainder.rfind(sep, max(0, int(last * .7)), last)
                      for sep in ("\n\n", "\n", ". ", "; ", " ")]
        split = next((p + len(sep) for p, sep in zip(candidates, ("\n\n", "\n", ". ", "; ", " ")) if p > 0), last)
        split = min(split, last)
        if split <= 0:
            split = last
        piece = remainder[:split]
        # Some tokenization boundaries are nonmonotone; verify and fallback.
        while split > 1 and len(tokenizer.encode(piece, add_special_tokens=False)) > max_tokens:
            split -= 1
            piece = remainder[:split]
        if not piece:
            raise RuntimeError("Failed to make progress splitting Markdown")
        segments.append(piece)
        index += len(piece)
    if "".join(segments) != text:
        raise AssertionError("Chunking changed source Markdown")
    return segments


def make_converter(args):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    p = PdfPipelineOptions()
    p.accelerator_options = AcceleratorOptions(num_threads=args.threads, device=args.device)
    p.do_ocr = False
    p.do_formula_enrichment = bool(args.formula)
    p.do_table_structure = True
    p.table_structure_options.mode = TableFormerMode.ACCURATE
    p.table_structure_options.do_cell_matching = True
    p.generate_picture_images = True
    p.generate_page_images = False
    p.images_scale = args.image_scale
    p.enable_remote_services = False
    # Supported in recent Docling; limits concurrent buffered pages.
    if hasattr(p, "queue_max_size"):
        p.queue_max_size = 1
    options = {"pipeline_options": p}
    selected_backend = getattr(args, "_pdf_backend", "threaded")
    if selected_backend == "pdfium":
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        options["backend"] = PyPdfiumDocumentBackend
    else:
        try:
            from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
            options["backend_options"] = ThreadedDoclingParseBackendOptions(max_concurrent_results=1)
        except ImportError:
            pass
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(**options)})


def extract_batch(pdf: Path, stage: Path, start: int, end: int, profile_id: str, args, converter=None, tokenizer=None):
    from docling_core.types.doc import ImageRefMode, PictureItem
    print(f"WORKER: {pdf.name}, source PDF pages {start}-{end}", flush=True)
    if converter is None:
        converter = make_converter(args)
    result = converter.convert(pdf, page_range=(start, end))
    status = str(getattr(result, "status", "unknown"))
    if "SUCCESS" not in status.upper() or "PARTIAL" in status.upper():
        raise RuntimeError(f"Docling status {status}; batch not checkpointed")
    doc = result.document
    if doc is None:
        raise RuntimeError("Docling returned no document")
    numbering = normalize_pages(doc, start, end)
    for sub in ("pages", "figures", "tables"):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    placeholder = "[FIGURE: see figures_to_review.jsonl]"
    if tokenizer is None:
        tokenizer = load_tokenizer()
    total_chunks = 0
    suspicious_pages = []
    with (stage / "batch.md").open("w", encoding="utf-8") as md, \
         (stage / "rag_chunks.jsonl").open("w", encoding="utf-8") as rag:
        for page in range(start, end + 1):
            content = doc.export_to_markdown(page_no=page,
                    image_mode=ImageRefMode.PLACEHOLDER,
                    image_placeholder=placeholder, traverse_pictures=True)
            (stage / "pages" / f"page_{page:05d}.md").write_text(
                f"# {pdf.name} — original PDF page {page}\n\n{content}\n", encoding="utf-8")
            md.write(f"\n\n---\n\n# Original PDF page {page}\n\n{content}\n")
            if len(content.strip()) < 20:
                suspicious_pages.append(page)
            fragments = chunk_by_tokens_exact(content, tokenizer, args.max_tokens)
            for n, piece in enumerate(fragments, 1):
                row = {"id": f"{pdf.stem}:p{page:05d}:c{n:04d}",
                       "book": pdf.name, "page": page, "pages": [page],
                       "text": piece,
                       "token_count": len(tokenizer.encode(piece, add_special_tokens=False)),
                       "chunk_type": "page_markdown", "fragment": n,
                       "fragments_on_page": len(fragments)}
                rag.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chunks += 1
    doc.save_as_json(stage / "structure.json", image_mode=ImageRefMode.PLACEHOLDER)

    tables = []
    for i, table in enumerate(doc.tables, 1):
        label = f"table_{i:04d}"
        errors = []
        csv_rel, html_rel = f"tables/{label}.csv", f"tables/{label}.html"
        try:
            table.export_to_dataframe(doc=doc).to_csv(stage / csv_rel, index=False, encoding="utf-8-sig")
        except Exception as exc:
            errors.append("csv export: " + repr(exc)); csv_rel = None
        try:
            (stage / html_rel).write_text(table.export_to_html(doc=doc), encoding="utf-8")
        except Exception as exc:
            errors.append("html export: " + repr(exc)); html_rel = None
        data = getattr(table, "data", None)
        rows, cols = getattr(data, "num_rows", None), getattr(data, "num_cols", None)
        if rows is not None and rows <= 1:
            errors.append("single detected row")
        if cols is not None and cols <= 1:
            errors.append("single detected column")
        try:
            caption = table.caption_text(doc=doc)
        except Exception as exc:
            caption = ""; errors.append("caption: " + repr(exc))
        tables.append({"id": f"p{start:05d}-{end:05d}:{label}", "book": pdf.name,
                       "pages": page_numbers(table), "caption": caption,
                       "rows": rows, "columns": cols,
                       "csv": csv_rel, "html": html_rel,
                       "flags": errors, "review_status": "needs_source_verification"})
    dump_jsonl(stage / "tables_index.jsonl", tables)

    figures = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        label = f"figure_{len(figures)+1:04d}"
        rel = f"figures/{label}.png"
        issue = None
        try:
            img = item.get_image(doc)
            if img is None:
                raise RuntimeError("No image returned from Docling")
            img.save(stage / rel, "PNG")
        except Exception as exc:
            rel = None; issue = repr(exc)
        try:
            caption = item.caption_text(doc=doc)
        except Exception:
            caption = ""
        figures.append({"id": f"p{start:05d}-{end:05d}:{label}", "book": pdf.name,
                        "pages": page_numbers(item), "caption": caption, "image": rel,
                        "description": "", "graph_data_csv": "",
                        "review_status": "needs_review", "extraction_error": issue})
    dump_jsonl(stage / "figures_to_review.jsonl", figures)
    warnings = []
    if suspicious_pages:
        warnings.append(f"Sparse text on pages: {suspicious_pages} (may be blank or figures)")
    if start > 1:
        warnings.append(f"Page {start} is a batch boundary; check continuing figures/tables/paragraphs")
    if end < args.total_pages:
        warnings.append(f"Page {end} is a batch boundary; check continuing figures/tables/paragraphs")
    warnings += [f"Table {t['id']}: {', '.join(t['flags'])}" for t in tables if t["flags"]]
    warnings += [f"Figure {f['id']}: {f['extraction_error']}" for f in figures if f["extraction_error"]]
    (stage / "review_flags.md").write_text(
        "# Batch extraction checks\n\n" +
        "\n".join("- " + w for w in warnings) +
        "\n\nImportant: numerical graphs, complex tables and equations must be source-verified.\n",
        encoding="utf-8")
    record = {
        "pipeline_version": VERSION, "profile_id": profile_id,
        "book": pdf.name, "start": start, "end": end,
        "page_numbering": numbering, "pages": end-start+1,
        "tables": len(tables), "figure_regions": len(figures), "chunks": total_chunks,
        "created_utc": now_utc(),
        "files": file_manifest(stage),
    }
    atomic_json(stage / MARKER, record)
    print(f"WORKER COMPLETE: {start}-{end}: {len(tables)} tables, "
          f"{len(figures)} figures, {total_chunks} chunks", flush=True)


def save_checkpoint_to_drive(stage: Path, checkpoint: Path, profile_id: str,
                             start: int, end: int):
    """Copy marker last, verify every file on Drive, then publish batch atomically."""
    valid, why = validate_checkpoint(stage, profile_id, start, end)
    if not valid:
        raise RuntimeError(f"Local batch staging invalid: {why}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    transfer = checkpoint.parent / (".copying_" + checkpoint.name)
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir()
    for item in stage.iterdir():
        if item.name == MARKER:
            continue
        if item.is_dir():
            shutil.copytree(item, transfer / item.name)
        else:
            shutil.copy2(item, transfer / item.name)
    shutil.copy2(stage / MARKER, transfer / MARKER)
    valid, why = validate_checkpoint(transfer, profile_id, start, end)
    if not valid:
        raise RuntimeError(f"Drive checkpoint copy failed verification: {why}")
    if checkpoint.exists():
        shutil.rmtree(checkpoint)  # verified invalid by parent, replaced here
    transfer.rename(checkpoint)
    print(f"SAVED checkpoint: pages {start}-{end}", flush=True)


def run_group_worker(a):
    """Load models once; process and persist each batch independently."""
    import gc
    tasks = json.loads(a._batch_list.read_text(encoding="utf-8"))
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA selected but no GPU runtime is available. "
                               "Choose Runtime > Change runtime type > GPU, then retry.")
    converter = make_converter(a)
    tokenizer = load_tokenizer()
    for start, end in tasks:
        checkpoint = a._dest_book_dir / "batches" / f"pages_{start:05d}_{end:05d}"
        ok, why = validate_checkpoint(checkpoint, a._profile_id, start, end)
        if ok:
            print(f"SKIP previously saved batch {start}-{end}", flush=True)
            continue
        stage = a._worker_dir / f"stage_pages_{start:05d}_{end:05d}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        print(f"CONVERT {a._pdf.name}: original pages {start}-{end}", flush=True)
        extract_batch(a._pdf, stage, start, end, a._profile_id, a,
                      converter=converter, tokenizer=tokenizer)
        backend_used = getattr(a, "_pdf_backend", "threaded")
        marker_path = stage / MARKER
        marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_data["pdf_backend"] = (
            "PyPdfiumDocumentBackend" if backend_used == "pdfium"
            else "ThreadedDoclingParseDocumentBackend")
        if backend_used == "pdfium":
            with (stage / "review_flags.md").open("a", encoding="utf-8") as f:
                f.write("\n- Alternative PyPdfium2 PDF backend used for this batch. "
                        "Verify tables, formulas, text order, and figures against the PDF.\n")
            marker_data["files"] = file_manifest(stage)
        atomic_json(marker_path, marker_data)
        save_checkpoint_to_drive(stage, checkpoint, a._profile_id, start, end)
        shutil.rmtree(stage)
        # Clear transient document/image allocations while retaining model weights.
        gc.collect()


def relative_assets(entry: dict, prefix: str) -> dict:
    result = dict(entry)
    for field in ("csv", "html", "image", "graph_data_csv"):
        if result.get(field):
            result[field] = prefix + result[field]
    return result


def assemble_book(book_dir: Path, pdf: Path, profile_id: str, ranges: list[tuple[int,int]]):
    staging = book_dir / ".assembled_partial"
    staging.mkdir(exist_ok=True)
    counts = {"pages": 0, "tables": 0, "figure_regions": 0, "chunks": 0}
    with (staging / "book.md").open("w", encoding="utf-8") as md, \
         (staging / "rag_chunks.jsonl").open("w", encoding="utf-8") as rag, \
         (staging / "tables_index.jsonl").open("w", encoding="utf-8") as tab, \
         (staging / "figures_to_review.jsonl").open("w", encoding="utf-8") as fig, \
         (staging / "quality_report.md").open("w", encoding="utf-8") as qa:
        md.write(f"# {pdf.name}\n\nExtracted from original PDF. PDF page numbers are physical, 1-based.\n")
        qa.write(f"# Review report: {pdf.name}\n\nEach batch has its own structure.json. "
                 "The book-level Markdown below is concatenated from page ranges; "
                 "chapter context and tables crossing batch boundaries need visual review.\n")
        for start, end in ranges:
            b = book_dir / "batches" / f"pages_{start:05d}_{end:05d}"
            checked, why = validate_checkpoint(b, profile_id, start, end)
            if not checked:
                raise RuntimeError(f"Checkpoint {b} changed during assembly: {why}")
            marker = json.loads((b / MARKER).read_text(encoding="utf-8"))
            counts["pages"] += marker["pages"]
            counts["chunks"] += marker["chunks"]
            counts["tables"] += marker["tables"]
            counts["figure_regions"] += marker["figure_regions"]
            md.write((b / "batch.md").read_text(encoding="utf-8"))
            with (b / "rag_chunks.jsonl").open(encoding="utf-8") as f:
                shutil.copyfileobj(f, rag)
            prefix = f"batches/{b.name}/"
            for filename, target in (("tables_index.jsonl", tab),( "figures_to_review.jsonl",fig)):
                with (b / filename).open(encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            target.write(json.dumps(relative_assets(json.loads(line), prefix), ensure_ascii=False) + "\n")
            qa.write(f"\n## Pages {start}-{end}\n\n")
            qa.write((b / "review_flags.md").read_text(encoding="utf-8"))
    targets = ("book.md", "rag_chunks.jsonl", "tables_index.jsonl", "figures_to_review.jsonl", "quality_report.md")
    for name in targets:
        os.replace(staging / name, book_dir / name)
    try:
        staging.rmdir()
    except OSError:
        pass
    atomic_json(book_dir / MARKER, {"profile_id": profile_id, "book": pdf.name,
                                    "ranges": ranges, **counts, "completed_utc": now_utc()})
    return counts


def assemble_global(out_root: Path, pdfs: list[Path]):
    """Create a retrievable index only from fully completed books."""
    tmp = out_root / "all_rag_chunks.jsonl.incomplete"
    final = out_root / "all_rag_chunks.jsonl"
    count = 0
    with tmp.open("w", encoding="utf-8") as dst:
        for pdf in pdfs:
            book_dir = out_root / pdf.stem
            if not (book_dir / MARKER).is_file() or not (book_dir / "rag_chunks.jsonl").is_file():
                continue
            with (book_dir / "rag_chunks.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        dst.write(line)
                        count += 1
    os.replace(tmp, final)
    return count


def argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Google Drive folder of PDFs")
    p.add_argument("--output", type=Path, required=True, help="Google Drive folder of V3 output")
    p.add_argument("--batch-pages", type=int, default=10, help="Page checkpoint size, default 10")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                   help="Docling inference device; defaults to cpu for V3 checkpoint compatibility")
    p.add_argument("--batches-per-worker", type=int, default=3,
                   help="Reuse models across N batches before restarting worker; default 3")
    p.add_argument("--max-tokens", type=int, default=220)
    p.add_argument("--image-scale", type=float, default=2.0)
    p.add_argument("--formula", action="store_true", help="Enable model-based formula recognition")
    p.add_argument("--only", nargs="*", help="PDF file name(s) or stem(s) to process")
    p.add_argument("--max-batches", type=int, default=None, help="Stop after N NEW batches (for testing)")
    p.add_argument("--backend-strategy", choices=("auto", "threaded", "pdfium"), default="auto",
                   help="auto: default parser, fallback on SIGTRAP; pdfium: use alternative parser "
                        "for new batches; threaded: original V3.1 behavior")
    # Deliberately excluded from the original extraction profile. Each batch records its actual PDF backend.
    p.add_argument("--_pdf_backend", choices=("threaded", "pdfium"), default="threaded", help=argparse.SUPPRESS)
    # Internal worker arguments; not for direct use by users.
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_pdf", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_stage", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_start", type=int, help=argparse.SUPPRESS)
    p.add_argument("--_end", type=int, help=argparse.SUPPRESS)
    p.add_argument("--_profile_id", type=str, help=argparse.SUPPRESS)
    p.add_argument("--_batch_list", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_dest_book_dir", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker_dir", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--total-pages", type=int, default=0, help=argparse.SUPPRESS)
    return p


def main():
    a = argparser().parse_args()
    if a.batch_pages < 1 or a.threads < 1 or a.batches_per_worker < 1 or not (32 <= a.max_tokens <= 240) or a.image_scale <= 0:
        raise SystemExit("Invalid settings: batch-pages/threads >=1, tokens 32..240, scale>0")
    if a._worker:
        if a._batch_list:
            run_group_worker(a)
        else:
            extract_batch(a._pdf, a._stage, a._start, a._end, a._profile_id, a)
        return
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available in this Colab runtime.")
    if not a.input.is_dir():
        raise SystemExit(f"Input folder not found: {a.input}")
    pdfs = sorted(a.input.glob("*.pdf"))
    if a.only:
        chosen = {x.casefold() for x in a.only}
        pdfs = [p for p in pdfs if p.stem.casefold() in chosen or p.name.casefold() in chosen]
    if not pdfs:
        raise SystemExit("No matching PDFs")
    a.output.mkdir(parents=True, exist_ok=True)
    completed_new = 0
    errors = []
    for pdf in pdfs:
        print(f"\n=== SOURCE: {pdf.name} ===", flush=True)
        # Local disk avoids repeatedly reading mounted Drive during conversion.
        with tempfile.TemporaryDirectory(prefix="pdf_v3_local_", dir="/content" if Path("/content").is_dir() else None) as temp:
            local = Path(temp) / pdf.name
            size = pdf.stat().st_size
            free = shutil.disk_usage(temp).free
            if free < size + (1 << 30):
                raise RuntimeError(f"Local storage insufficient; need PDF size plus 1 GiB free, available {free}")
            print("Copying PDF to Colab temporary storage...", flush=True)
            shutil.copy2(pdf, local)
            source_sha = sha_file(local)
            import pypdfium2 as pdfium
            src_doc = pdfium.PdfDocument(str(local))
            try:
                total = len(src_doc)
            finally:
                src_doc.close()
            profile = make_profile(local, source_sha, a)
            profile_id = fingerprint(profile)
            book_dir = a.output / pdf.stem
            book_dir.mkdir(parents=True, exist_ok=True)
            saved_profile = book_dir / "pipeline_profile.json"
            if saved_profile.exists():
                older = json.loads(saved_profile.read_text(encoding="utf-8"))
                if older != profile:
                    keys = [k for k in profile if older.get(k) != profile[k]]
                    raise RuntimeError(f"Checkpoint settings/source changed for {pdf.name}: {keys}. "
                                       "Keep previous settings or use a DIFFERENT output directory.")
            else:
                if any(book_dir.iterdir()):
                    raise RuntimeError(f"Existing files in {book_dir} without V3 profile; "
                                       "choose an unused output directory")
                atomic_json(saved_profile, profile)
            ranges = [(s, min(s + a.batch_pages - 1, total)) for s in range(1, total + 1, a.batch_pages)]
            pending = []
            for start, end in ranges:
                checkpoint = book_dir / "batches" / f"pages_{start:05d}_{end:05d}"
                ok, why = validate_checkpoint(checkpoint, profile_id, start, end)
                print(f"{'SKIP' if ok else 'QUEUE'} {pdf.name}: pages {start}-{end}: {why}", flush=True)
                if not ok:
                    pending.append((start, end))
            # Process remaining batches, preserving already verified V3.1 checkpoints.
            # A SIGTRAP kills the *worker*, not this parent process, so fallback
            # must run as a separate worker. Preserve the original profile ID.
            remaining = list(pending)
            while remaining:
                if a.max_batches is not None and completed_new >= a.max_batches:
                    print("Maximum new batches reached; restart with the same settings to resume.")
                    return
                slots = a.batches_per_worker
                if a.max_batches is not None:
                    slots = min(slots, a.max_batches - completed_new)
                group = remaining[:slots]
                if not group:
                    break

                def run_worker_group(jobs, pdf_backend):
                    batch_list = Path(temp) / "batch_jobs.json"
                    batch_list.write_text(json.dumps(jobs), encoding="utf-8")
                    command = [sys.executable, str(Path(__file__).resolve()),
                               "--input", str(a.input), "--output", str(a.output),
                               "--batch-pages", str(a.batch_pages),
                               "--threads", str(a.threads), "--device", a.device,
                               "--batches-per-worker", str(a.batches_per_worker),
                               "--max-tokens", str(a.max_tokens),
                               "--image-scale", str(a.image_scale),
                               "--total-pages", str(total),
                               "--_worker", "--_pdf", str(local),
                               "--_profile_id", profile_id,
                               "--_batch_list", str(batch_list),
                               "--_dest_book_dir", str(book_dir),
                               "--_worker_dir", str(temp),
                               "--_pdf_backend", pdf_backend]
                    if a.formula:
                        command.append("--formula")
                    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads),
                               MKL_NUM_THREADS=str(a.threads),
                               OPENBLAS_NUM_THREADS=str(a.threads),
                               TOKENIZERS_PARALLELISM="false")
                    print(f"WORKER GROUP [{pdf_backend}]: {len(jobs)} batches, pages "
                          f"{jobs[0][0]}-{jobs[-1][1]}", flush=True)
                    return subprocess.run(command, env=env, check=False).returncode

                primary = "pdfium" if a.backend_strategy == "pdfium" else "threaded"
                rc = run_worker_group(group, primary)
                succeeded = []
                for start, end in group:
                    ckpt = book_dir / "batches" / f"pages_{start:05d}_{end:05d}"
                    valid, _why = validate_checkpoint(ckpt, profile_id, start, end)
                    if valid:
                        succeeded.append((start, end))
                completed_new += len(succeeded)
                remaining = [x for x in remaining if x not in succeeded]
                if rc == 0 and len(succeeded) == len(group):
                    continue

                if a.backend_strategy == "auto" and rc == -5:
                    # Salvage the first failed batch only. Retry subsequent batches
                    # with the default backend, rather than silently changing them.
                    failed = next((b for b in group if b not in succeeded), None)
                    if failed is None:
                        # Signal observed after the entire group was persisted.
                        continue
                    print(f"SIGTRAP: retrying pages {failed[0]}-{failed[1]} "
                          "using PyPdfium2 without touching earlier checkpoints", flush=True)
                    rc2 = run_worker_group([failed], "pdfium")
                    path = book_dir / "batches" / f"pages_{failed[0]:05d}_{failed[1]:05d}"
                    recovered, reason = validate_checkpoint(path, profile_id, *failed)
                    if rc2 != 0 or not recovered:
                        raise RuntimeError(f"Alternative parser also failed for pages {failed}: "
                                           f"exit={rc2}, checkpoint={reason}")
                    completed_new += 1
                    remaining.remove(failed)
                    print(f"AUTO-RECOVERED pages {failed[0]}-{failed[1]}", flush=True)
                    continue

                raise RuntimeError(f"Worker group incomplete (exit {rc}); "
                                   f"{len(succeeded)}/{len(group)} batches saved; "
                                   "run again with identical settings or test --backend-strategy pdfium.")
            if all(validate_checkpoint(book_dir / "batches" / f"pages_{s:05d}_{e:05d}", profile_id, s, e)[0]
                   for s, e in ranges):
                summary = assemble_book(book_dir, pdf, profile_id, ranges)
                print(f"BOOK COMPLETE: {pdf.name}, {total} pages, "
                      f"{summary['tables']} tables, {summary['figure_regions']} figures, "
                      f"{summary['chunks']} chunks", flush=True)
    combined = assemble_global(a.output, pdfs)
    print(f"Combined {combined} chunks from fully assembled books.")
    print(f"\nCompleted this session: {completed_new} new batches; failures: {len(errors)}")
    print("All processed books can be found under:", a.output)


if __name__ == "__main__":
    main()
