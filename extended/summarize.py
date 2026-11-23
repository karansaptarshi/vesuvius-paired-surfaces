"""Summarize registration at two withheld landmarks per extended site, including failed sites."""
import json
from pathlib import Path
import numpy as np

def summary(folder):
    records=[]
    for p in sorted(folder.glob('*/result.json')):
        res=json.loads(p.read_text())
        row={'site':p.parent.name,'status':res['status']}
        a=p.parent/'anchors.json'
        if a.exists():
            d=json.loads(a.read_text());v=np.asarray(d['shift_low_zyx']);xy=np.asarray(d['locations_yx'])
            if v.shape==(6,3) and xy.shape==(6,2):
                X=np.c_[np.ones(6),(xy-384)/216];coef=np.linalg.lstsq(X[:4],v[:4],rcond=None)[0]
                before=np.linalg.norm(v[4:],axis=1);after=np.linalg.norm((X@coef-v)[4:],axis=1)
                if 'heldout_errors_low_voxels' in res:np.testing.assert_allclose(after,res['heldout_errors_low_voxels'])
                row.update(global_um=(9.362*before).tolist(),corrected_um=(9.362*after).tolist())
        records.append(row)
    ready=[r for r in records if r['status']=='paired scans ready for visual reading']
    cat=lambda subset,key:np.array([x for rec in subset for x in rec.get(key,[])])
    def metrics(rows):
        a=cat(rows,'global_um');b=cat(rows,'corrected_um')
        if not len(a):return {}
        return {'sites':len(rows),'withheld_landmarks_with_shifts':len(a),'median_global_um':float(np.median(a)),'median_corrected_um':float(np.median(b)),'corrected_below_global':int((a>b).sum())}
    out={'selection':'Two withheld points per attempted site, after fitting four points. All completed sites and the subset passing the independent match and error checks are listed separately. At failed registrations the estimated landmark shifts may themselves be unreliable.','all_recorded_sites':metrics(records),'verified_sites':metrics(ready),'sites':records}
    (folder/'heldout-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out['all_recorded_sites'],out['verified_sites'])
if __name__=='__main__':summary(Path(__file__).resolve().parent)
