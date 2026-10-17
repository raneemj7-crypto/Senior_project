import json
from pathlib import Path
import tempfile
import unittest
from run_sequence_cv import CANDIDATES, select_candidate, validate_cache, digest


class PlanningTests(unittest.TestCase):
    def test_development_only(self):
        dev=['MOT20-02','MOT20-03','MOT20-05']
        rows=[dict(sequence=n,candidate=c,idf1=.5) for n in dev for c in CANDIDATES]
        rows.append(dict(sequence='MOT20-01',candidate='match07',idf1=1.0))
        chosen,_=select_candidate(rows,dev)
        self.assertEqual(chosen,'baseline')
        rows[1]['idf1']=.8
        self.assertEqual(select_candidate(rows,dev)[0],'buffer60')

    def test_incomplete_development_rejected(self):
        with self.assertRaises(ValueError): select_candidate([],['MOT20-01'])

    def test_cache_empty_frames_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); signature={'frames':2}
            path=root/'detections.jsonl'
            path.write_text(json.dumps(dict(frame=1,detections=[]))+'\n'+json.dumps(dict(frame=2,detections=[[1,2,4,5,.8,0]]))+'\n')
            (root/'metadata.json').write_text(json.dumps(dict(signature=signature,complete=True,sha256=digest(path))))
            self.assertEqual(validate_cache(root,signature),path)
            path.write_text('')
            with self.assertRaises(ValueError): validate_cache(root,signature)


if __name__=='__main__': unittest.main()
