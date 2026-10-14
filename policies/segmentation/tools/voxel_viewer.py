#!/usr/bin/env python3
"""Interactive 3D point cloud viewer — orbit the scene while the video plays.
Every depth pixel is back-projected to 3D and colored by segmentation class.
Usage: python3 tools/voxel_viewer.py [--camera front-wide] [--depth-scale 7.0]
"""
import numpy as np
import os
import struct
import re
import time
import threading
import webbrowser
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from functools import lru_cache
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive.paths import DATA_DIR, DEPTH_DIR, load_calib

SKIP_SKY = 10
M_TO_FT = 3.28084

_g = {}

HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>3D Point Cloud Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;color:#ccc;font-family:'SF Mono',Menlo,monospace;overflow:hidden}
#app{display:flex;flex-direction:column;height:100vh}
#main{display:flex;flex:1;min-height:0}
#vid-panel{flex:0 0 42%;display:flex;align-items:center;justify-content:center;background:#080808;border-right:1px solid #222}
video{max-width:100%;max-height:100%}
#gl-panel{flex:1;position:relative}
#bar{height:42px;display:flex;align-items:center;gap:10px;padding:0 14px;background:#141414;border-top:1px solid #222}
input[type=range]{flex:1;height:4px;-webkit-appearance:none;appearance:none;background:#333;border-radius:2px;cursor:pointer;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#4af;cursor:pointer}
button{background:#252525;color:#ccc;border:1px solid #3a3a3a;padding:3px 10px;border-radius:4px;cursor:pointer;font:12px/1 monospace}
button:hover{background:#333}
#hint{position:absolute;top:6px;left:8px;font-size:10px;color:#555;pointer-events:none}
#stats{position:absolute;bottom:6px;right:8px;font-size:11px;color:#555;pointer-events:none}
#tm{font-size:12px;color:#888;min-width:55px}
#spd{font-size:11px;color:#666;min-width:24px;text-align:center}
#legend{position:absolute;top:6px;right:8px;font-size:10px;line-height:16px;color:#888;pointer-events:none}
.lc{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:middle}
</style></head><body>
<div id="app">
 <div id="main">
  <div id="vid-panel"><video id="vid" src="/video" preload="auto" muted></video></div>
  <div id="gl-panel">
   <div id="hint">Drag to orbit &middot; Scroll to zoom &middot; Shift+drag to pan &middot; [ ] point size</div>
   <div id="legend">
    <div><span class="lc" style="background:rgb(128,64,128)"></span>road</div>
    <div><span class="lc" style="background:rgb(244,35,232)"></span>sidewalk</div>
    <div><span class="lc" style="background:rgb(70,70,70)"></span>building</div>
    <div><span class="lc" style="background:rgb(107,142,35)"></span>vegetation</div>
    <div><span class="lc" style="background:rgb(152,251,152)"></span>terrain</div>
    <div><span class="lc" style="background:rgb(220,20,60)"></span>person</div>
    <div><span class="lc" style="background:rgb(0,0,142)"></span>car</div>
    <div><span class="lc" style="background:rgb(0,0,70)"></span>truck</div>
   </div>
   <div id="stats"></div>
  </div>
 </div>
 <div id="bar">
  <button id="pbtn">&#9654;</button>
  <span id="tm">0:00</span>
  <input type="range" id="seek" min="0" max="__NUM_FRAMES__" value="0">
  <button id="sl">&minus;</button>
  <span id="spd">1&times;</span>
  <button id="fa">+</button>
  <button id="rc">Reset Cam</button>
 </div>
</div>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const NF=__NUM_FRAMES__, FPS=30, CH=__CAM_H_FT__;
const MAX_PTS=__MAX_PTS__;

const COLORS_F=[
  [0.502,0.251,0.502],[0.957,0.137,0.910],[0.275,0.275,0.275],[0.400,0.400,0.612],[0.745,0.600,0.600],
  [0.600,0.600,0.600],[0.980,0.667,0.118],[0.863,0.863,0.000],[0.420,0.557,0.137],[0.596,0.984,0.596],
  [0.275,0.510,0.706],[0.863,0.078,0.235],[1.000,0.000,0.000],[0.000,0.000,0.557],[0.000,0.000,0.275],
  [0.000,0.235,0.392],[0.000,0.314,0.392],[0.000,0.000,0.902],[0.467,0.043,0.125]
];

const wrap=document.getElementById('gl-panel');
const vid=document.getElementById('vid');
const seek=document.getElementById('seek');
const stats=document.getElementById('stats');

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0a0a12);

const cam=new THREE.PerspectiveCamera(55,wrap.clientWidth/wrap.clientHeight,1,600);
cam.position.set(0,25,15);

const ren=new THREE.WebGLRenderer({antialias:false});
ren.setSize(wrap.clientWidth,wrap.clientHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2));
wrap.appendChild(ren.domElement);

const ctrl=new OrbitControls(cam,ren.domElement);
ctrl.target.set(0,3,-60);
ctrl.enableDamping=true;
ctrl.dampingFactor=0.08;
ctrl.update();

// point cloud
const geo=new THREE.BufferGeometry();
const posArr=new Float32Array(MAX_PTS*3);
const colArr=new Float32Array(MAX_PTS*3);
geo.setAttribute('position',new THREE.BufferAttribute(posArr,3));
geo.setAttribute('color',new THREE.BufferAttribute(colArr,3));
geo.setDrawRange(0,0);

const mat=new THREE.PointsMaterial({size:0.4,vertexColors:true,sizeAttenuation:true});
const pts=new THREE.Points(geo,mat);
pts.frustumCulled=false;
scene.add(pts);

// ego arrow
const ego=new THREE.Mesh(
  new THREE.ConeGeometry(1,2.5,4).rotateX(Math.PI/2),
  new THREE.MeshBasicMaterial({color:0x3399ff}));
ego.position.set(0,CH-0.3,0);
scene.add(ego);

// labeled ground grid in feet
const GY=CH;
const gridC=new THREE.LineBasicMaterial({color:0x223322});
const gridCFaint=new THREE.LineBasicMaterial({color:0x181e18});
function mkLine(p,m){const g=new THREE.BufferGeometry().setFromPoints(p);scene.add(new THREE.Line(g,m));}

// forward lines every 25ft, from 0 to 200ft
for(let d=0;d<=200;d+=25){
  const z=-d;
  mkLine([new THREE.Vector3(-50,GY,z),new THREE.Vector3(50,GY,z)], d%50===0?gridC:gridCFaint);
}
// lateral lines every 25ft, from -50 to +50
for(let x=-50;x<=50;x+=25){
  mkLine([new THREE.Vector3(x,GY,0),new THREE.Vector3(x,GY,-200)], x===0?gridC:gridCFaint);
}

// text labels via canvas sprites
function mkLabel(text,pos,align,size){
  const c=document.createElement('canvas');c.width=192;c.height=48;
  const ctx=c.getContext('2d');
  ctx.font='bold 26px monospace';ctx.fillStyle='#667766';
  ctx.textAlign=align||'left';ctx.textBaseline='middle';
  ctx.fillText(text,align==='right'?190:2,24);
  const tex=new THREE.CanvasTexture(c);tex.minFilter=THREE.LinearFilter;
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));
  sp.position.copy(pos);const s=size||1;sp.scale.set(12*s,3*s,1);
  scene.add(sp);
}
// forward distance labels
for(let d=25;d<=200;d+=25){
  mkLabel(d+"'",new THREE.Vector3(53,GY,-d),'left');
}
// lateral labels
for(let x=-50;x<=50;x+=25){
  if(x===0) mkLabel("0",new THREE.Vector3(x,GY,5),'center');
  else mkLabel((x>0?'+':'')+x+"'",new THREE.Vector3(x,GY,5),'center');
}
mkLabel('forward',new THREE.Vector3(55,GY+2,-100),'left',1.2);
mkLabel('lateral',new THREE.Vector3(0,GY+2,10),'center',1.2);

// frame management
const cache={};
const pending=new Set();
let cur=-1,lastUp=0;

async function ff(i){
  if(cache[i]||pending.has(i))return;
  if(pending.size>=6)return;
  pending.add(i);
  try{const r=await fetch('/frame/'+i);cache[i]=await r.arrayBuffer();}
  catch(e){}
  finally{pending.delete(i);}
}

function apply(buf){
  const dv=new DataView(buf);
  const n=Math.min(dv.getUint32(0,true),MAX_PTS);
  const pA=geo.attributes.position.array;
  const cA=geo.attributes.color.array;

  for(let i=0;i<n;i++){
    const o=4+i*7;
    // int16 centifeet -> feet
    const wx=dv.getInt16(o,true)/100.0;
    const wy=dv.getInt16(o+2,true)/100.0;
    const wz=dv.getInt16(o+4,true)/100.0;
    const cl=dv.getUint8(o+6);
    const p=i*3;
    pA[p]=wx;
    pA[p+1]=-wy;
    pA[p+2]=-wz;
    const rgb=COLORS_F[cl]||[0.5,0.5,0.5];
    cA[p]=rgb[0];cA[p+1]=rgb[1];cA[p+2]=rgb[2];
  }
  geo.setDrawRange(0,n);
  geo.attributes.position.needsUpdate=true;
  geo.attributes.color.needsUpdate=true;
  stats.textContent=n.toLocaleString()+' points';
}

function tick(){
  requestAnimationFrame(tick);
  ctrl.update();
  const now=performance.now();
  if(now-lastUp>55){
    lastUp=now;
    const fi=Math.max(0,Math.min(Math.floor(vid.currentTime*FPS),NF-1));
    if(fi!==cur){
      cur=fi;
      seek.value=fi;
      const s=vid.currentTime;
      document.getElementById('tm').textContent=
        Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');
      if(cache[fi])apply(cache[fi]);
      else ff(fi).then(()=>{if(cache[fi]&&cur===fi)apply(cache[fi]);});
      for(let j=1;j<=20;j++){const p=fi+j;if(p<NF)ff(p);}
    }
  }
  ren.render(scene,cam);
}
tick();
for(let i=0;i<30;i++)ff(i);

let speed=1;
document.getElementById('pbtn').onclick=()=>{
  if(vid.paused){vid.play();document.getElementById('pbtn').innerHTML='&#9646;&#9646;';}
  else{vid.pause();document.getElementById('pbtn').innerHTML='&#9654;';}
};
seek.oninput=()=>{vid.currentTime=seek.value/FPS;cur=-1;};
document.getElementById('sl').onclick=()=>{speed=Math.max(0.25,speed/2);vid.playbackRate=speed;document.getElementById('spd').textContent=speed+'×';};
document.getElementById('fa').onclick=()=>{speed=Math.min(4,speed*2);vid.playbackRate=speed;document.getElementById('spd').textContent=speed+'×';};
document.getElementById('rc').onclick=()=>{cam.position.set(0,25,15);ctrl.target.set(0,3,-60);ctrl.update();};
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();document.getElementById('pbtn').click();}
  if(e.code==='ArrowLeft'){vid.currentTime=Math.max(0,vid.currentTime-(e.shiftKey?5:1/FPS));cur=-1;}
  if(e.code==='ArrowRight'){vid.currentTime=Math.min(vid.duration||9999,vid.currentTime+(e.shiftKey?5:1/FPS));cur=-1;}
  if(e.code==='BracketRight'){mat.size=Math.min(3.0,mat.size*1.2);}
  if(e.code==='BracketLeft'){mat.size=Math.max(0.05,mat.size/1.2);}
});
window.addEventListener('resize',()=>{
  cam.aspect=wrap.clientWidth/wrap.clientHeight;
  cam.updateProjectionMatrix();
  ren.setSize(wrap.clientWidth,wrap.clientHeight);
});
vid.addEventListener('loadedmetadata',()=>{seek.max=Math.floor(vid.duration*FPS);});
</script></body></html>"""


def init_data(camera_name, depth_scale):
    data_dir = str(DATA_DIR)
    depth_dir = str(DEPTH_DIR / camera_name)
    calib = load_calib()

    da3_d = os.path.join(depth_dir, "da3_depth_maps.npz")
    da3_s = os.path.join(depth_dir, "da3_seg_maps.npz")
    v2_d = os.path.join(depth_dir, "depth_maps.npz")
    v2_s = os.path.join(depth_dir, "seg_maps_30fps.npz")

    if os.path.exists(da3_d) and os.path.exists(da3_s):
        dp, sp = da3_d, da3_s
        print("Using DA3 depth + seg")
    elif os.path.exists(v2_d) and os.path.exists(v2_s):
        dp, sp = v2_d, v2_s
        print("Using V2 depth + seg")
    else:
        raise FileNotFoundError(f"No depth/seg data in {depth_dir}")

    print(f"Depth scale factor: {depth_scale}x", flush=True)

    print("Loading depth maps...", flush=True)
    t0 = time.time()
    d = np.load(dp)
    _g['depth_maps'] = d['depth_maps']
    _g['depth_scales'] = d['depth_scales']
    print(f"  {_g['depth_maps'].shape} in {time.time()-t0:.1f}s", flush=True)

    print("Loading seg maps...", flush=True)
    t0 = time.time()
    s = np.load(sp)
    _g['seg_maps'] = s['seg_maps']
    print(f"  {_g['seg_maps'].shape} in {time.time()-t0:.1f}s", flush=True)

    _g['num_frames'] = min(len(_g['depth_maps']), len(_g['seg_maps']))
    _g['video_path'] = os.path.join(data_dir, f"{camera_name}.mp4")
    if not os.path.exists(_g['video_path']):
        raise FileNotFoundError(f"Video not found: {_g['video_path']}")

    _g['cam_h_m'] = calib['extrinsics']['height_m']
    _g['cam_h_ft'] = _g['cam_h_m'] * M_TO_FT
    _g['depth_scale'] = depth_scale

    dh, dw = _g['depth_maps'].shape[1], _g['depth_maps'].shape[2]
    vid_h, vid_w = _g['seg_maps'].shape[1], _g['seg_maps'].shape[2]
    _g['seg_ds_y'] = vid_h // dh
    _g['seg_ds_x'] = vid_w // dw

    focal = calib['intrinsics']['focal_length']
    cx = calib['intrinsics']['cx']
    cy = calib['intrinsics']['cy']
    pitch = np.radians(calib['extrinsics']['pitch_deg'])
    cp, sp_ = np.cos(pitch), np.sin(pitch)

    u = np.arange(dw) * (vid_w / dw) + (vid_w / dw) / 2
    v = np.arange(dh) * (vid_h / dh) + (vid_h / dh) / 2
    uu, vv = np.meshgrid(u, v)
    ray_x = (uu - cx) / focal
    ray_y = (vv - cy) / focal

    _g['rx'] = ray_x.astype(np.float32)
    _g['ry'] = (cp * ray_y + sp_).astype(np.float32)
    _g['rz'] = (-sp_ * ray_y + cp).astype(np.float32)

    _g['max_pts'] = dh * dw

    max_depth_ft = _g['depth_scales'].max() * depth_scale * M_TO_FT
    print(f"Ready: {_g['num_frames']} frames, max depth ~{max_depth_ft:.0f}ft", flush=True)


@lru_cache(maxsize=2048)
def compute_frame(idx):
    if idx < 0 or idx >= _g['num_frames']:
        return struct.pack('<I', 0)

    depth_raw = _g['depth_maps'][idx].astype(np.float32) * (_g['depth_scales'][idx] / 65535.0)
    seg = _g['seg_maps'][idx]
    seg_d = seg[::_g['seg_ds_y'], ::_g['seg_ds_x']]
    dh, dw = depth_raw.shape
    seg_d = seg_d[:dh, :dw]

    depth = depth_raw * _g['depth_scale']

    px = _g['rx'] * depth
    py = _g['ry'] * depth - _g['cam_h_m']
    pz = _g['rz'] * depth

    sc = np.clip(seg_d, 0, 18)
    valid = (depth_raw > 0.1) & (sc != SKIP_SKY)

    wx = px[valid] * M_TO_FT
    wy = py[valid] * M_TO_FT
    wz = pz[valid] * M_TO_FT
    wc = sc[valid]

    n = len(wx)
    if n == 0:
        return struct.pack('<I', 0)

    # Pack as int16 centifeet (0.01ft precision) + uint8 class = 7 bytes/pt
    x_cf = (wx * 100).clip(-32768, 32767).astype(np.int16)
    y_cf = (wy * 100).clip(-32768, 32767).astype(np.int16)
    z_cf = (wz * 100).clip(-32768, 32767).astype(np.int16)

    buf = np.empty(n * 7, dtype=np.uint8)
    buf[0::7] = x_cf.view(np.uint8)[0::2]
    buf[1::7] = x_cf.view(np.uint8)[1::2]
    buf[2::7] = y_cf.view(np.uint8)[0::2]
    buf[3::7] = y_cf.view(np.uint8)[1::2]
    buf[4::7] = z_cf.view(np.uint8)[0::2]
    buf[5::7] = z_cf.view(np.uint8)[1::2]
    buf[6::7] = wc.astype(np.uint8)

    return struct.pack('<I', n) + buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split('?')[0]
        try:
            if p == '/':
                self._html()
            elif p.startswith('/frame/'):
                self._frame(int(p.split('/')[-1]))
            elif p == '/video':
                self._video()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _html(self):
        html = HTML
        reps = {
            '__NUM_FRAMES__': str(_g['num_frames']),
            '__CAM_H_FT__': f"{_g['cam_h_ft']:.2f}",
            '__MAX_PTS__': str(_g['max_pts']),
        }
        for k, v in reps.items():
            html = html.replace(k, v)
        data = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def _frame(self, idx):
        data = compute_frame(idx)
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', len(data))
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()
        self.wfile.write(data)

    def _video(self):
        fp = _g['video_path']
        sz = os.path.getsize(fp)
        rng = self.headers.get('Range')
        if rng:
            m = re.match(r'bytes=(\d+)-(\d*)', rng)
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else sz - 1
            length = end - start + 1
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{sz}')
            self.send_header('Content-Length', str(length))
        else:
            start, length = 0, sz
            self.send_response(200)
            self.send_header('Content-Length', str(sz))
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        with open(fp, 'rb') as f:
            f.seek(start)
            rem = length
            while rem > 0:
                chunk = f.read(min(65536, rem))
                if not chunk:
                    break
                self.wfile.write(chunk)
                rem -= len(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive 3D point cloud viewer')
    parser.add_argument('--camera', default='front-wide',
                        choices=['front-wide', 'front-narrow'])
    parser.add_argument('--depth-scale', type=float, default=7.0,
                        help='Multiply DA3 depth by this factor (default 7.0)')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    init_data(args.camera, args.depth_scale)

    server = ThreadingHTTPServer(('localhost', args.port), Handler)
    url = f'http://localhost:{args.port}'
    print(f'\nServing at {url}')
    print('Press Ctrl+C to stop\n')

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
        server.shutdown()
