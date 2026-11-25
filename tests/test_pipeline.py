import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json_to_ai as j
import convert_books_v3_1_autorecover as engine

class PipelineTests(unittest.TestCase):
    def test_lexical_roundtrip(self):
        raw = r'{"x":1.00e-3,"x":-0,"s":"\u03b1\n```"}'
        self.assertEqual(j.tokens(raw), j.tokens(j.pretty_json(raw)))
    def test_nested_repeat_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); out = p / 'AI_KNOWLEDGE'
            (p/'data.jsonl').write_text('{"x":1}\n\n{"x":2}\n')
            a = j.convert(p,out,dry_run=False)
            b = j.convert(p,out,dry_run=False)
            self.assertEqual(a['run'],b['run'])
            self.assertEqual(a['files'][0]['records'],2)
            previous=(out/'LATEST.json').read_bytes()
            (p/'bad.jsonl').write_text('{"x":1}\ninvalid\n')
            with self.assertRaises(RuntimeError): j.convert(p,out,dry_run=False)
            self.assertEqual(previous,(out/'LATEST.json').read_bytes())
    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'pages').mkdir()
            for name in ('batch.md','rag_chunks.jsonl','structure.json','tables_index.jsonl','figures_to_review.jsonl','pages/page_1.md'):
                (p/name).write_text('test')
            engine.atomic_json(p/'_COMPLETE.json',{'profile_id':'test','start':1,'end':1,'files':engine.file_manifest(p)})
            self.assertTrue(engine.validate_checkpoint(p,'test',1,1)[0])
            (p/'batch.md').write_text('changed')
            self.assertFalse(engine.validate_checkpoint(p,'test',1,1)[0])
