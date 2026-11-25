"""Read-only integrity audit for V3.1 extraction outputs."""
import json
from pathlib import Path
import convert_books_v3_1_autorecover as engine

def audit_book(book, pdf_input):
    book=Path(book); profile=json.loads((book/'pipeline_profile.json').read_text()); marker=json.loads((book/'_COMPLETE.json').read_text())
    pid=engine.fingerprint(profile); issues=[]
    source=Path(pdf_input)/profile['source_filename']
    if not source.is_file(): issues.append('Original source unavailable')
    elif engine.sha_file(source)!=profile['source_sha256']: issues.append('Source SHA mismatch')
    else:
        try:
            import pypdfium2
            doc=pypdfium2.PdfDocument(str(source)); total=len(doc); doc.close()
        except ImportError: total=None; issues.append('PDFium not installed; source page count unverified')
    ranges=marker['ranges']; expected=1
    if marker.get('profile_id')!=pid: issues.append('Root marker profile mismatch')
    for start,end in ranges:
        if start!=expected or end<start: issues.append(f'Noncontiguous range {start}-{end}')
        expected=end+1
        batch=book/'batches'/f'pages_{start:05d}_{end:05d}'
        ok,reason=engine.validate_checkpoint(batch,pid,start,end)
        if not ok: issues.append(f'{batch.name}: {reason}')
    if source.is_file() and 'total' in locals() and total is not None and total!=expected-1: issues.append('Source page count differs')
    for name in ('rag_chunks.jsonl','tables_index.jsonl','figures_to_review.jsonl'):
        try:
            merged=[]
            for start,end in ranges:
                batch=book/'batches'/f'pages_{start:05d}_{end:05d}'
                for line in (batch/name).read_text().splitlines():
                    if line.strip():
                        row=json.loads(line)
                        merged.append(engine.relative_assets(row,f'batches/{batch.name}/') if name!='rag_chunks.jsonl' else row)
            actual=[json.loads(l) for l in (book/name).read_text().splitlines() if l.strip()]
            print(name, 'root:',len(actual),'batch:',len(merged),'equal:',actual==merged)
            if actual!=merged: issues.append(name+': different aggregation; retain both and inspect')
            for row in actual:
                for key in ('csv','html','image','graph_data_csv'):
                    if row.get(key) and not (book/row[key]).is_file(): issues.append('Missing asset: '+str(row[key]))
        except (OSError,ValueError,KeyError) as e: issues.append(f'{name}: {e}')
    print('Physical pages from ranges:',expected-1,'; issues:',len(issues))
    for issue in issues: print('WARNING:',issue)
    return issues
