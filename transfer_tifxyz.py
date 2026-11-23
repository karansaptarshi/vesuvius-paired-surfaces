"""Transfer a listed-scan TIFXYZ surface to the corresponding partial rescan."""
import argparse
import json
from pathlib import Path
import uuid
import numpy as np
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz
from map_points import Registration

def write_surface(dest,zyx,valid,scale,scan,description,source):
    if not np.any(valid):
        raise ValueError("No surface vertices overlap the chosen volume")
    coords=[np.where(valid,a,-1).astype('float32') for a in zyx[::-1]]
    bbox=tuple(float(a[valid].min()) for a in coords)+tuple(float(a[valid].max()) for a in coords)
    ident=str(uuid.uuid5(uuid.NAMESPACE_URL,scan+'#'+description+'#'+source))
    surf=Tifxyz(*coords,uuid=ident,_scale=tuple(scale),bbox=bbox,_mask=valid,
                extra={'target_volume':scan.rstrip('/').split('/')[-1],'volume_url':scan,
                       'description':description,'source_surface':source})
    return write_tifxyz(dest,surf,compression='deflate',tile_size=128)

def transfer(scroll,source,dest,pilot_grid=None):
    cal=Registration(scroll)
    info=json.loads((cal.folder/'coordinates.json').read_text())
    surface=read_tifxyz(source,validate=True,load_mask=True)
    declared=surface.extra.get('volume_url')
    if declared and declared.rstrip('/')!=info['listed']['zarr'].rstrip('/'):
        raise ValueError('The surface declares a different source CT volume')
    x,y,z,valid=surface[:,:];valid=valid.copy()
    lowcoords=np.stack([z,y,x]).astype('float64')
    valid&=np.isfinite(lowcoords).all(axis=0)
    for v,n in zip(lowcoords,cal.listed_shape):valid&=(v>=0)&(v<=n-1)
    coordinates=lowcoords
    if pilot_grid is not None:
        sy,sx,dy,dx=pilot_grid
        if not np.isfinite(pilot_grid).all() or min(dy,dx)<=0:
            raise ValueError("Pilot origin and positive grid steps must be finite")
        ij=np.indices(valid.shape,dtype=float)
        py=sy+dy*ij[0];px=sx+dx*ij[1]
        # Local calibration belongs to the original named sheet. Compare the
        # world-space positions with that trace before extending its correction.
        from scipy.ndimage import map_coordinates
        with np.load(cal.folder/'listed-sheet.npz') as data:
            sheet=data['zyx'];mask=data['valid'].astype(float)
            span=(py>=0)&(px>=0)&(py<sheet.shape[1])&(px<sheet.shape[2])
            uv=[py,px]
            distance=np.sqrt(sum((map_coordinates(a,uv,order=1,mode='nearest')-b)**2 for a,b in zip(sheet,lowcoords)))
            mask_ok=map_coordinates(mask,uv,order=1,mode='nearest')>.999
            if np.any(valid&(~span|~mask_ok|(distance>2.0))):
                raise ValueError("Pilot grid does not match the supplied reference sheet at the specified atlas positions")
        design=np.stack([np.ones_like(py),(py-384)/224,(px-384)/224])
        coordinates=lowcoords+np.einsum('ij,i...->j...',cal.fit,design)
    high=np.einsum('ij,j...->i...',cal.matrix,coordinates)+cal.translation[:,None,None]
    for v,n in zip(high,cal.rescan_shape):valid&=(v>=0)&(v<=n-1)
    scale=tuple(np.asarray(surface._scale)*info['rescan']['voxel_um']/info['listed']['voxel_um'])
    mode='locally calibrated named trace' if pilot_grid is not None else 'global affine registration'
    write_surface(dest,high,valid,scale,info['rescan']['zarr'],scroll+' '+mode,surface.uuid)
    return int(valid.sum())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scroll',choices=('PHerc0846A','PHerc1203'));p.add_argument('source',type=Path);p.add_argument('dest',type=Path)
    p.add_argument('--pilot-grid',type=float,nargs=4,metavar=('ROW0','COL0','ROW_STEP','COL_STEP'),help='use only for a donor grid aligned to the archived pilot trace; units are 9.6 micrometer atlas pixels')
    a=p.parse_args();print(transfer(a.scroll,a.source,a.dest,a.pilot_grid),'valid target vertices')
