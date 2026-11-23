"""Read a paired surface with the official TIFXYZ reader and sample its original public CT volume."""
import argparse,json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from PIL import Image
from vesuvius.tifxyz import read_tifxyz
from recheck_anchor import sample

def render(surface_dir,output,level=0,size=32):
    if not 1<=size<=96:raise ValueError("Select from 1 to 96 stored vertices per side for this public-volume preview")
    surf=read_tifxyz(surface_dir,load_mask=True,validate=True)
    meta=json.loads((surface_dir/'meta.json').read_text());url=meta['volume_url']+f'{level}/'
    with urlopen(url+'.zarray',timeout=45) as f:vm=json.load(f)
    h,w=surf.shape;sz=min(size,h,w);y=h//2-sz//2;x=w//2-sz//2
    sx,sy,szcoord,valid=surf[y:y+sz,x:x+sz];coords=np.stack([szcoord,sy,sx]).astype('float64')
    if not valid.any():raise ValueError("No valid points in center window")
    coords[:,~valid]=coords[:,valid][:,:1]
    pixels=sample(url,((coords+.5)/2**level-.5)[:,None,:,:],vm)[0]
    pixels=np.rint(np.clip(pixels,0,255)).astype('uint8');pixels[~valid]=0
    Image.fromarray(pixels).save(output)
    return pixels,[y,x]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('surface',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--level',type=int,choices=range(6),default=0);p.add_argument('--size',type=int,default=32)
    a=p.parse_args();pix,origin=render(a.surface,a.output,a.level,a.size);print('stored-grid origin',origin,'shape',pix.shape)
