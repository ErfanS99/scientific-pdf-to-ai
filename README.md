# Scientific PDF to AI

A modular research-document extraction pipeline for converting born-digital engineering PDFs into page-cited Markdown, structured data, visual assets and retrieval chunks. An independent JSON/JSONL publisher converts existing structured data to verified Markdown without rerunning extraction.

## Quick start — Google Colab

1. Clone or download this repository. Copy `src/` into `MyDrive/PDF_AI_Project/` (or change `SRC` in the notebook to your clone). Open `Scientific_PDF_to_AI.ipynb` in Colab.
2. Choose a T4 GPU runtime for PDF extraction. Upload your own PDFs to `MyDrive/PDF_AI_Project/pdfs/`. The notebook creates input/output folders if missing.
3. Mount Drive, set `INSTALL_PDF_DEPS=True` for a fresh PDF runtime, and run the library-loading section.
4. Set `ONLY` to a list of filenames (empty means all), and `RUN_PDF=True`. Run the extraction section. Preview mode is the default.
5. For existing JSON data, use `RUN_JSON=True`, select `JSON_INPUT`, and run the JSON section. GPU and Docling are not required for JSON conversion.

## Command line

Python 3.10+ recommended. Install PDF dependencies with `python -m pip install -r requirements.txt`; use a CUDA-compatible PyTorch installation for GPU use. Docling and tokenizer models require network access on first use. Dependency ranges are not a fully pinned reproducible environment; resume with the exact saved Docling version and extraction profile.

```bash
python -u src/convert_books_v3_1_autorecover.py \
  --input pdfs --output knowledge --batch-pages 10 --threads 8 \
  --device cuda --batches-per-worker 3 --max-tokens 220 \
  --image-scale 2.0 --formula --backend-strategy pdfium
```

This is the best observed T4 configuration from the project, not a universal benchmark. Use `--only Your_Book.pdf` to select one source. Use `--max-batches 1` with a separate test output directory for a smoke test. For CPU use `--device cpu` and a separate output directory if switching from CUDA. Use `--help` for all options.

```python
import sys
sys.path.insert(0, 'src')
from json_to_ai import convert
convert('knowledge/Your_Book', 'knowledge/Your_Book/AI_KNOWLEDGE', dry_run=True)
# After reviewing the plan, repeat with dry_run=False.
```

## Repository layout

- `Scientific_PDF_to_AI.ipynb`: short operational guide and configuration interface.
- `src/convert_books_v3_1_autorecover.py`: original V3.1 engine, preserving its checkpoint schema and recovery behavior.
- `src/json_to_ai.py`: standard-library JSON/JSONL converter, lexical readback verification and deterministic publication manifests.
- `src/audit_extraction.py`: read-only source/checkpoint and aggregation audit.
- `pdfs/`: empty input placeholder; never commit source PDFs.
- `tests/`: CPU-only regression tests; no model downloads.

## Output and scientific fidelity

Each book has `book.md`, `quality_report.md`, `rag_chunks.jsonl`, table and figure indexes, and batch directories containing page Markdown, figures, tables, structure JSON and completion markers. Preserve original PDFs separately for verification. Resume with identical source bytes, filename, Docling version and profile. A selected-book run may rebuild the global RAG index with selected books only; use per-book indexes and back up manually edited publications. OCR is disabled; scanned documents are not a supported fidelity target. Complex equations, numerical charts and cross-page tables require manual source verification. Checksums are integrity evidence, not proof of scientific accuracy.

JSON publication preserves lexical tokens (including duplicate keys and exact number spellings), not original whitespace. Default output is Markdown, with optional TXT. Invalid records prevent promotion of the latest successful run. The default excludes operational metadata and detailed structure; opt in if needed. Do not suppress batch indexes without checking equivalence. Native Google Docs publication is not implemented.

## Tests

```bash
python -m unittest discover -s tests -v
```

CPU-only tests cover JSON formatting/publication and checkpoint integrity. End-to-end Colab GPU execution remains to be validated for this repository release.

## License and dependencies

MIT (see `LICENSE`), proposed for the project-owned source code. Third-party libraries and model weights retain their respective licenses. No research PDFs, extracted research data, credentials or model weights are distributed. Use only documents you are authorized to process.
