"""Display predicted log-depth with the frozen training codec, never as GT."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
NEAR, FAR = .01, 2.0

def decode(value):
    value=np.asarray(value,dtype=np.float32)
    if not np.isfinite(value).all() or np.any((value<0)|(value>1)):
        raise ValueError('normalized depth out of range')
    return np.exp((1-value)*np.log(FAR/NEAR)+np.log(NEAR)).astype(np.float32), value>0

def render(depth_path):
    source=Path(depth_path); base=source.parent
    index=int(source.stem.split('-')[-1])
    data=np.load(base/'predicted-depth.npy',allow_pickle=False)
    if data.ndim!=5 or data.shape[:1]!=(1,) or data.shape[1] not in (1,3) or data.shape[2]!=12:
        raise ValueError('unexpected depth tensor')
    value=data[0,:,index].mean(axis=0)
    meters,valid=decode(value)
    maskpath=base/'depth-content-mask.png'
    if maskpath.exists():
        mask=np.asarray(Image.open(maskpath))
    else:
        # Frozen service field-map: centered 480x426 content on 480x640.
        if value.shape != (480,640):raise ValueError('unverified camera packing')
        mask=np.zeros((480,640),np.uint8);mask[:,107:533]=255
    if mask.shape!=value.shape:raise ValueError('depth mask shape mismatch')
    valid &= mask>0
    # Keep original grayscale encoding; add a fixed meter legend, no per-frame rescale.
    gray=np.rint(value*255).astype(np.uint8)
    rgb=np.repeat(gray[:,:,None],3,axis=2);rgb[~valid]=[180,0,180]
    h,w=gray.shape
    canvas=Image.new('RGB',(w,h+100),'white');canvas.paste(Image.fromarray(rgb),(0,0))
    draw=ImageDraw.Draw(canvas)
    draw.text((8,h+3),'PREDICTED depth (m); log scale; NOT measured',fill='black')
    x0,x1=20,w-20
    for x in range(x0,x1+1):
        g=int(round(255*(1-(x-x0)/(x1-x0))))
        draw.line((x,h+24,x,h+39),fill=(g,g,g))
    for z in (.01,.05,.1,.25,.5,1.,2.):
        x=x0+int(np.log(z/NEAR)/np.log(FAR/NEAR)*(x1-x0))
        draw.line((x,h+39,x,h+45),fill='black');draw.text((min(w-32,max(0,x-10)),h+47),str(z),fill='black')
    draw.text((8,h+72),'Magenta: padding / zero-code (invalid or far endpoint)',fill='black')
    out=base/f'metric-depth-{index:03d}.png';canvas.save(out)
    np.save(base/f'metric-depth-{index:03d}.npy',np.where(valid,meters,np.nan),allow_pickle=False)
    (base/'metric-depth-codec.json').write_text(json.dumps(dict(mode='log',near_m=NEAR,far_m=FAR,source='predicted-depth.npy',invalid='zero-code or content-mask',display='fixed log grayscale meter legend',measured=False)))
    return str(out)
