"""Repair existing monitor compare.mp4 padding; preserve originals and metrics.

Usage: python scripts/mask_robotwin_depth_panels.py --round-dir RUN/eval/step_0004000 --backup-dir BACKUP
Only robotwin_3d 640x480, 2-row/3-column panels are accepted. This is a display
repair of compressed previews, not a regeneration of model predictions/metrics.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.data_prepare.robotwin_3d import motus_t_view_slots


def main():
    import cv2
    import imageio_ffmpeg
    ap = argparse.ArgumentParser()
    ap.add_argument('--round-dir', type=Path, required=True)
    ap.add_argument('--backup-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.backup_dir.exists():
        raise FileExistsError('Use a fresh backup directory; repair must not run twice')
    paths = sorted(a.round_dir.glob('rank*/sample*/compare.mp4'))
    if not paths:
        raise FileNotFoundError(a.round_dir)
    a.backup_dir.mkdir(parents=True)
    h, w = 480, 640
    r0, r1, c0, c1 = motus_t_view_slots((h,w))['content']
    assert (r0,r1)==(0,h)
    filters = []
    for col in range(3):
        for x, width in [(col*w,c0),(col*w+c1,w-c1)]:
            filters.append(f'drawbox=x={x}:y={h}:w={width}:h={h}:color=black:t=fill')
    records=[]
    for p in paths:
        meta=json.loads(p.with_name('meta.json').read_text())
        if meta.get('source')!='robotwin_3d':
            raise ValueError(f'Unexpected source: {p}')
        cap=cv2.VideoCapture(str(p)); shape=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if shape!=(2*h,3*w):
            raise ValueError(f'Unexpected panel shape {shape}: {p}')
        backup=a.backup_dir/p.relative_to(a.round_dir);backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,backup)
        temp=p.with_name('compare.padding_tmp.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-threads','1','-i',str(backup),
                        '-vf',','.join(filters),'-c:v','libx264','-threads','2','-preset','fast','-crf','18',
                        '-pix_fmt','yuv420p','-an',str(temp)],check=True)
        cap=cv2.VideoCapture(str(temp));ok,frame=cap.read();newcount=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if not ok or newcount!=count or frame.shape[:2]!=shape:
            raise RuntimeError(f'Failed output validation: {p}')
        os.replace(temp,p)
        records.append(dict(path=str(p),frames=count,fps=fps))
        if len(records)%32==0:print('Repaired',len(records),flush=True)
    (a.backup_dir/'repair_manifest.json').write_text(json.dumps(records,indent=2))
    print('Repaired',len(records),'previews; original metrics unchanged',flush=True)

if __name__=='__main__':main()
