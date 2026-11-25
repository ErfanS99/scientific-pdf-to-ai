"""Lossless lexical JSON/JSONL publication; never changes source files."""
from pathlib import Path
import hashlib, json, re, os, tempfile
from urllib.parse import quote

VERSION = '1.0'
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[^\s{}\[\],:"]+|[{}\[\],:]')
EXCLUDE = {'json_output','ai_knowledge','_project','archive','original_extraction','admin','.git'} - {'original_extraction'}
ADMIN = {'_complete.json','pipeline_profile.json','manifest.json','published_docs.json','failures.jsonl'}
def sha(b): return hashlib.sha256(b).hexdigest()
def tokens(s): return TOKEN.findall(s)
def reject_constant(s): raise ValueError('Nonstandard JSON constant: '+s)
def parse(s): return json.loads(s, parse_int=str, parse_float=str, parse_constant=reject_constant)
def pretty_json(s):
    parse(s); ts=tokens(s); out=[]; depth=0
    for i,t in enumerate(ts):
        prev=ts[i-1] if i else None
        if t in ('}',']'):
            depth-=1
            if prev not in ('{','['): out.append('\n'+'  '*depth)
            out.append(t)
        elif t in ('{','['):
            out.append(t); depth+=1
            if i+1<len(ts) and ts[i+1] not in ('}',']'): out.append('\n'+'  '*depth)
        elif t==',': out.append(',\n'+'  '*depth)
        elif t==':': out.append(': ')
        else: out.append(t)
    result=''.join(out)+'\n'; parse(result)
    if tokens(result)!=ts: raise ValueError('Token mismatch')
    return result

def atomic(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.tmp-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data)
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def discover(source, output, detailed=False, include_admin=False, canonical=True):
    source=Path(source).resolve(); output=Path(output).resolve()
    if not source.is_dir(): raise FileNotFoundError(source)
    if source==output or output in source.parents: raise ValueError('Output must be separate or nested, not an ancestor')
    found=[]
    for root,dirs,files in os.walk(source,followlinks=False):
        root=Path(root)
        dirs[:]=sorted(d for d in dirs if d.casefold() not in EXCLUDE and not d.startswith('.') and not (root/d).is_symlink() and (root/d).resolve()!=output)
        for name in sorted(files):
            p=root/name; rel=p.relative_to(source)
            if p.is_symlink() or p.suffix.lower() not in {'.json','.jsonl','.ndjson'}: continue
            if not include_admin and name.casefold() in ADMIN: continue
            if not detailed and name.casefold()=='structure.json': continue
            # Canonical mode skips batch-level index duplicates only after explicit user choice.
            if canonical and 'batches' in rel.parts and name in {'rag_chunks.jsonl','tables_index.jsonl','figures_to_review.jsonl'}:
                book=source.joinpath(*rel.parts[:rel.parts.index('batches')]);
                if (book/name).is_file(): continue
            found.append(p)
    return sorted(found)

def render(raw, rel, suffix):
    s=raw.decode('utf-8-sig'); records=[]
    if suffix=='.json': records=[(1,s)]
    else: records=[(i,line) for i,line in enumerate(s.splitlines(),1) if line.strip()]
    if not records: raise ValueError('No JSON records')
    chunks=[]; lexical=0
    for line,record in records:
        try: pretty=pretty_json(record)
        except Exception as exc: raise ValueError(f'Invalid record at source line {line}: {exc}') from exc
        fence='`'*max(3,1+max(map(len,re.findall(r'`+',pretty)),default=0))
        chunks.append((line,record,pretty,fence)); lexical+=len(tokens(record))
    header=f'# {rel.name}\n\nSource: `{rel.as_posix()}`\n\nSHA-256: `{sha(raw)}`\n\n'
    body=''.join(f'## Source line {line}\n\n{fence}json\n{pretty}{fence}\n\n' for line,_,pretty,fence in chunks)
    return (header+body).encode(),chunks,lexical

def verify(data,chunks):
    text=data.decode('utf-8'); pos=0
    for line,original,_,fence in chunks:
        start=f'## Source line {line}\n\n{fence}json\n'; a=text.index(start,pos)+len(start); b=text.index('\n'+fence+'\n',a)
        extracted=text[a:b]; parse(extracted)
        if tokens(extracted)!=tokens(original): raise ValueError(f'Readback mismatch at line {line}')
        pos=b+len(fence)+2
    if text[pos:].strip(): raise ValueError('Unexpected trailing data')

def convert(source,output,*,detailed=False,include_admin=False,canonical=False,write_txt=False,dry_run=True):
    source=Path(source).resolve(); output=Path(output).resolve()
    paths=discover(source,output,detailed,include_admin,canonical)
    if dry_run:
        for p in paths: print('PLAN',p.relative_to(source))
        return {'status':'dry_run','count':len(paths)}
    if not paths: raise ValueError('No matching inputs')
    # Content-addressed immutable runs; canonical pointer is written only after all files verify.
    inputs=[(p,p.relative_to(source),sha(p.read_bytes())) for p in paths]
    config={'version':VERSION,'detailed':detailed,'include_admin':include_admin,'canonical':canonical,'write_txt':write_txt}
    run_id=sha(json.dumps([config,[(str(r),h) for _,r,h in inputs]],sort_keys=True).encode())[:24]
    run=output/('run_'+run_id); rows=[]
    for p,rel,h in inputs:
        row={'source':str(rel),'source_sha256':h}
        try:
            raw=p.read_bytes()
            if sha(raw)!=h: raise ValueError('Source changed after inventory')
            data,chunks,count=render(raw,rel,p.suffix.lower())
            dest=run/'documents'/rel.parent/(rel.name+'.md')
            if not dest.exists() or dest.read_bytes()!=data: atomic(dest,data)
            saved=dest.read_bytes(); verify(saved,chunks)
            row.update(status='verified',source_bytes=len(raw),output=str(dest.relative_to(run)),output_sha256=sha(saved),output_bytes=len(saved),records=len(chunks),lexical_tokens=count)
            if write_txt:
                txt=dest.with_suffix('.txt'); content=''.join(pretty for _,_,pretty,_ in chunks).encode(); atomic(txt,content)
                if txt.read_bytes()!=content: raise ValueError('TXT readback mismatch')
                row['txt_sha256']=sha(content)
        except Exception as exc: row.update(status='failed',error=str(exc))
        rows.append(row); print(row['status'],rel)
    complete=all(r['status']=='verified' for r in rows)
    manifest={'config':config,'status':'verified' if complete else 'partial','run':run_id,'files':rows}
    atomic(run/'manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False).encode())
    index=['# JSON publication index','',f'Status: {manifest["status"]}','','Original source files remain authoritative. Lexical token counts are not LLM token counts.','']
    for r in rows:
        index.append(f'- [{r["source"]}]({quote(r["output"],safe="/")}) — verified' if r['status']=='verified' else f'- {r["source"]} — FAILED: {r["error"]}')
    atomic(run/'INDEX.md',('\n'.join(index)+'\n').encode())
    if not complete: raise RuntimeError(f'Partial conversion; see {run}/manifest.json. Latest successful pointer unchanged.')
    atomic(output/'LATEST.json',json.dumps({'run':run.name,'index':run.name+'/INDEX.md'},indent=2).encode())
    print('Verified:',run); return manifest
