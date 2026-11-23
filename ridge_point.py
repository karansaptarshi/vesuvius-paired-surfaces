"""Read the supplied PHerc. 846A adjacent-layer lattice in both scan coordinates."""
import argparse,json
from pathlib import Path
import numpy as np

def lookup(layer,y,x):
    stem=Path(__file__).resolve().parent/'neighbor-sheet'/f'{layer}.npz'
    with np.load(stem,allow_pickle=False) as z:
        axis=z['atlas_axis']
        if not np.isfinite([y,x]).all() or min(y,x)<axis[0] or max(y,x)>axis[-1]:
            raise ValueError(f'Both atlas coordinates must be within {axis[0]} to {axis[-1]}')
        y0=min(int(y)//4,len(axis)-2);x0=min(int(x)//4,len(axis)-2)
        fy=(y-axis[y0])/4;fx=(x-axis[x0])/4
        weighted=[(y0,x0,(1-fy)*(1-fx)),(y0+1,x0,fy*(1-fx)),(y0,x0+1,(1-fy)*fx),(y0+1,x0+1,fy*fx)]
        active=[t for t in weighted if t[2]>0]
        if any(not z['valid'][i,j] for i,j,_ in active):raise ValueError('The selected position intersects the masked atlas region')
        result={'layer':layer,'atlas_yx':[y,x],'interpolated':len(active)>1}
        for key in ['depth_um','listed_zyx','rescan_zyx']:
            a=z[key]
            result[key]=np.asarray(sum(w*a[...,i,j].astype('f8') for i,j,w in active)).tolist()
        return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('layer',choices=['positive','centre','negative']);p.add_argument('y',type=float);p.add_argument('x',type=float)
    a=p.parse_args();print(json.dumps(lookup(a.layer,a.y,a.x),indent=2))
