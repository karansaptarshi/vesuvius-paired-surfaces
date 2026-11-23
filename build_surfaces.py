"""Export the two pilot traces and the three PHerc. 846A density ridges as paired TIFXYZ surfaces."""
import json
from pathlib import Path
import numpy as np
from vesuvius.tifxyz import read_tifxyz
from map_points import Registration
from transfer_tifxyz import write_surface,transfer
def build(output):
    root=Path(__file__).resolve().parent
    output.mkdir(parents=True,exist_ok=True)
    entries=[]
    def check(path,coords,valid,info,scan,name):
        surf=read_tifxyz(path,validate=True,load_mask=True);x,y,z,m=surf[:,:]
        assert np.array_equal(m,valid)
        tolerance=max(float(abs(a[m]-b[m]).max()) for a,b in zip([z,y,x],coords))
        assert tolerance<.002
        meta=json.loads((path/'meta.json').read_text());assert meta['type']=='seg' and meta['format']=='tifxyz'
        assert np.max(abs(np.array(meta['scale'])-info[scan]['voxel_um']/38.4))<1e-7
        assert all((c[m]>=0).all() and (c[m]<n).all() for c,n in zip([z,y,x],info[scan]['shape_zyx']))
        entries.append({'scroll':info['scroll'],'surface':name,'scan':scan,'path':str(path.relative_to(output)),'volume':info[scan]['zarr'],'vertices':int(m.sum()),'maximum_coordinate_difference_voxels':tolerance,'nominal_sampling_um':38.4})
    for scroll in ('PHerc0846A','PHerc1203'):
        cal=Registration(scroll);info=json.loads((cal.folder/'coordinates.json').read_text());info['scroll']=scroll
        d=output/scroll;d.mkdir(exist_ok=True)
        with np.load(cal.folder/'listed-sheet.npz') as f:
            points=f['zyx'][:,::4,::4];mask=f['valid'][::4,::4]
        l=d/'trace-listed';h=d/'trace-rescan'
        write_surface(l,points,mask,[info['listed']['voxel_um']/38.4]*2,info['listed']['zarr'],scroll+' reference pilot trace',info['mesh'])
        check(l,points,mask,info,'listed','trace')
        n=transfer(scroll,l,h,(0.,0.,4.,4.));assert n==int(mask.sum())
        ij=4*np.indices(mask.shape);design=np.stack([np.ones_like(ij[0]),(ij[0]-384)/224,(ij[1]-384)/224])
        high=np.einsum('ij,j...->i...',cal.matrix,points+np.einsum('ki,k...->i...',cal.fit,design))+cal.translation[:,None,None]
        check(h,high,mask,info,'rescan','trace')
        for y,x in [(64,64),(96,96),(128,128)]:
            if mask[y,x]:
                a,b=cal.surface_point((4*y,4*x));assert np.max(abs(a-points[:,y,x]))<.001 and np.max(abs(b-high[:,y,x]))<.001
        if scroll!='PHerc0846A':continue
        for layer in ('positive','centre','negative'):
            with np.load(root/'neighbor-sheet'/(layer+'.npz')) as f:
                for scan in ('listed','rescan'):
                    dst=d/(layer+'-'+scan);coords=f[scan+'_zyx'];mask=f['valid']
                    write_surface(dst,coords,mask,[info[scan]['voxel_um']/38.4]*2,info[scan]['zarr'],scroll+' CT ridge near '+layer,info['mesh'])
                    check(dst,coords,mask,info,scan,layer)
    (output/'index.json').write_text(json.dumps({'upstream_reader':'vesuvius.tifxyz.read_tifxyz with its validation and mask loading enabled','entries':entries},indent=2)+'\n')
    print('Verified',len(entries),'surfaces and',sum(e['vertices'] for e in entries),'valid vertices')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();build(a.output)
