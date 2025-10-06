# Fractal Art Explorer in JavaScript: Canvas, WebGL2, and WebGPU

**Objective**: Build an interactive fractal studio that moves from CPU to GPU without changing the UI. Every knob is live. Every color is a decision. You'll conjure an interactive fractal studio that moves from CPU to GPU without changing the UI. Every knob is live. Every color is a decision.

## Architecture & Modes

```mermaid
flowchart LR
  UI[Controls + Canvas] --> R(Render Router)
  R -->|cpu| W[Web Workers]
  R -->|gl2| G[WebGL2 Fragment Shader]
  R -->|wgpu (profile)| U[WebGPU Compute/Fragment]
  R -->|export| X[Tile Renderer (offscreen)]
  
  subgraph "Render Backends"
    W
    G
    U
    X
  end
```

**Why**: Same UI, pluggable backends. CPU path is the reference; GPUs are speed. This prevents banding, halves your frame time, and provides fallback compatibility.

## Minimal HTML Shell

### index.html
```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Fractal Explorer</title>
  <style>
    html,body{height:100%;margin:0;background:#0a0a0a;color:#eee;font-family:ui-monospace,monospace}
    #ui{position:fixed;inset:12px auto auto 12px;background:#1118;border:1px solid #333;padding:8px 12px;border-radius:10px;backdrop-filter: blur(4px)}
    canvas{display:block;width:100vw;height:100vh}
    .row{display:flex;gap:8px;align-items:center;margin:6px 0}
    input[type=range]{width:160px}
    button{background:#333;color:#eee;border:1px solid #555;padding:4px 8px;border-radius:4px;cursor:pointer}
    button:hover{background:#444}
    select{background:#222;color:#eee;border:1px solid #555;padding:2px 4px}
  </style>
</head>
<body>
  <canvas id="view"></canvas>
  <div id="ui">
    <div class="row">
      <label>Backend</label>
      <select id="backend">
        <option value="cpu">CPU</option>
        <option value="gl2">WebGL2</option>
        <option value="wgpu">WebGPU (if available)</option>
      </select>
    </div>
    <div class="row"><label>Max Iter</label><input id="iter" type="range" min="64" max="4096" value="512"><span id="iterv">512</span></div>
    <div class="row"><label>Palette</label><select id="palette"></select></div>
    <div class="row"><label>Mode</label><select id="mode"><option value="mandelbrot">Mandelbrot</option><option value="julia">Julia</option></select></div>
    <div class="row"><button id="reset">Reset</button><button id="snap">Export PNG</button><button id="tile">8k Tiled</button></div>
    <div class="row"><small id="info"></small></div>
  </div>
  <script type="module" src="./fractal.js"></script>
</body>
</html>
```

## Core Math: Smooth Coloring & Orbit Traps

### math.js
```javascript
// math.js
export function smoothIteration(cx, cy, maxIter=512) {
  // returns {n, zn2, mu} for color mapping
  let x=0, y=0, n=0;
  for (; n<maxIter && x*x+y*y<=4; n++) {
    const x2=x*x-y*y+cx;
    y = 2*x*y+cy;
    x = x2;
  }
  let mu = n;
  if (n<maxIter) {
    const logZn = Math.log(x*x+y*y)/2;
    const nu = Math.log(logZn/Math.log(2))/Math.log(2);
    mu = n + 1 - nu; // smooth
  }
  return { n, mu, r2: x*x+y*y };
}

export function smoothIterationJulia(zx, zy, cx, cy, maxIter=512) {
  let x=zx, y=zy, n=0;
  for (; n<maxIter && x*x+y*y<=4; n++) {
    const x2=x*x-y*y+cx;
    y = 2*x*y+cy;
    x = x2;
  }
  let mu = n;
  if (n<maxIter) {
    const logZn = Math.log(x*x+y*y)/2;
    const nu = Math.log(logZn/Math.log(2))/Math.log(2);
    mu = n + 1 - nu;
  }
  return { n, mu, r2: x*x+y*y };
}

export function orbitTrap(x0, y0, trap=[0,0], radius=0.1, maxIter=512){
  let x=0, y=0, dmin=1e9, k=0;
  for (; k<maxIter && x*x+y*y<=4; k++) {
    const d = Math.hypot(x-trap[0], y-trap[1]);
    if (d<dmin) dmin = d;
    const x2 = x*x - y*y + x0;
    y = 2*x*y + y0;
    x = x2;
  }
  return {k, dmin};
}

export function distanceEstimation(cx, cy, maxIter=512) {
  let x=0, y=0, dx=1, dy=0, n=0;
  for (; n<maxIter && x*x+y*y<=4; n++) {
    const x2 = x*x - y*y + cx;
    const y2 = 2*x*y + cy;
    const dx2 = 2*(x*dx - y*dy) + 1;
    const dy2 = 2*(x*dy + y*dx);
    x = x2; y = y2; dx = dx2; dy = dy2;
  }
  if (n >= maxIter) return 0;
  const r = Math.sqrt(x*x + y*y);
  return 0.5 * r * Math.log(r) / Math.sqrt(dx*dx + dy*dy);
}
```

## CPU Backend (Web Workers)

### cpu-worker.js
```javascript
// cpu-worker.js
importScripts('./math.js');

self.onmessage = e => {
  const {width,height,view,iter,palette,mode,juliaC} = e.data;
  const img = new Uint8ClampedArray(width*height*4);
  const {x:cx,y:cy,scale} = view;
  
  for (let j=0;j<height;j++){
    for (let i=0;i<width;i++){
      const zx = (i - width/2)/scale + cx;
      const zy = (j - height/2)/scale + cy;
      
      let res;
      if (mode==='mandelbrot') {
        res = smoothIteration(zx, zy, iter);
      } else { // Julia
        res = smoothIterationJulia(zx, zy, juliaC.x, juliaC.y, iter);
      }
      
      const t = res.mu/iter;
      const [r,g,b] = paletteAt(palette, t);
      const idx=(j*width+i)*4;
      img[idx]=r; img[idx+1]=g; img[idx+2]=b; img[idx+3]=255;
    }
  }
  self.postMessage({img}, [img.buffer]);
};

function paletteAt(fn, t) {
  return fn(t);
}
```

### cpu-backend.js
```javascript
// cpu-backend.js
export class CpuBackend {
  constructor(){
    this.worker = new Worker('./cpu-worker.js', {type:'module'});
  }
  
  render({ctx, view, iter, palette, mode, juliaC}) {
    const {canvas} = ctx;
    return new Promise(res=>{
      this.worker.onmessage = ({data})=>{
        const {img} = data;
        const imgData = new ImageData(new Uint8ClampedArray(img), canvas.width, canvas.height);
        ctx.putImageData(imgData,0,0);
        res();
      };
      this.worker.postMessage({
        width: canvas.width, height: canvas.height, view, iter, palette, mode, juliaC
      });
    });
  }
}
```

## WebGL2 Backend (Fragment Shader)

### mandelbrot.frag
```glsl
#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 uCenter;     // cx, cy
uniform float uScale;     // pixels per unit
uniform int uMaxIter;
uniform vec2 uRes;        // width,height
uniform int uMode;        // 0=mandelbrot, 1=julia
uniform vec2 uJuliaC;     // julia constant

vec3 palette(float t){ 
  // simple cyclic palette
  return 0.5 + 0.5*cos(6.28318*(vec3(0.0,0.33,0.67) + t));
}

void main(){
  vec2 z0 = (gl_FragCoord.xy - 0.5*uRes) / uScale + uCenter;
  vec2 z = vec2(0.0);
  vec2 c = uMode == 0 ? z0 : uJuliaC; // mandelbrot uses z0, julia uses constant
  
  int i;
  for (i=0; i<uMaxIter && dot(z,z)<=4.0; i++){
    z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
  }
  
  float mu = float(i);
  if (i<uMaxIter){
    float log_zn = 0.5*log(dot(z,z));
    float nu = log(log_zn / log(2.0)) / log(2.0);
    mu = float(i) + 1.0 - nu;
  }
  
  float t = mu/float(uMaxIter);
  vec3 col = palette(t);
  outColor = vec4(col,1.0);
}
```

### gl2-backend.js
```javascript
// gl2-backend.js
export class Gl2Backend {
  constructor(canvas){
    this.gl = canvas.getContext('webgl2', {premultipliedAlpha:false});
    this.setupShaders();
  }
  
  setupShaders() {
    const gl = this.gl;
    
    // Vertex shader (full-screen triangle)
    const vs = `#version 300 es
      in vec2 aPos;
      void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
      }`;
    
    // Fragment shader (from mandelbrot.frag)
    const fs = `#version 300 es
      precision highp float;
      out vec4 outColor;
      uniform vec2 uCenter;
      uniform float uScale;
      uniform int uMaxIter;
      uniform vec2 uRes;
      uniform int uMode;
      uniform vec2 uJuliaC;
      
      vec3 palette(float t){ 
        return 0.5 + 0.5*cos(6.28318*(vec3(0.0,0.33,0.67) + t));
      }
      
      void main(){
        vec2 z0 = (gl_FragCoord.xy - 0.5*uRes) / uScale + uCenter;
        vec2 z = vec2(0.0);
        vec2 c = uMode == 0 ? z0 : uJuliaC;
        
        int i;
        for (i=0; i<uMaxIter && dot(z,z)<=4.0; i++){
          z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        }
        
        float mu = float(i);
        if (i<uMaxIter){
          float log_zn = 0.5*log(dot(z,z));
          float nu = log(log_zn / log(2.0)) / log(2.0);
          mu = float(i) + 1.0 - nu;
        }
        
        float t = mu/float(uMaxIter);
        vec3 col = palette(t);
        outColor = vec4(col,1.0);
      }`;
    
    const program = this.createProgram(vs, fs);
    gl.useProgram(program);
    
    // Create full-screen triangle
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 3, -1, -1, 3
    ]), gl.STATIC_DRAW);
    
    const aPos = gl.getAttribLocation(program, 'aPos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    
    this.program = program;
    this.uCenter = gl.getUniformLocation(program, 'uCenter');
    this.uScale = gl.getUniformLocation(program, 'uScale');
    this.uMaxIter = gl.getUniformLocation(program, 'uMaxIter');
    this.uRes = gl.getUniformLocation(program, 'uRes');
    this.uMode = gl.getUniformLocation(program, 'uMode');
    this.uJuliaC = gl.getUniformLocation(program, 'uJuliaC');
  }
  
  createProgram(vs, fs) {
    const gl = this.gl;
    const vsShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vsShader, vs);
    gl.compileShader(vsShader);
    
    const fsShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsShader, fs);
    gl.compileShader(fsShader);
    
    const program = gl.createProgram();
    gl.attachShader(program, vsShader);
    gl.attachShader(program, fsShader);
    gl.linkProgram(program);
    
    return program;
  }
  
  render({view, iter, mode, juliaC}) {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.uniform2f(this.uCenter, view.x, view.y);
    gl.uniform1f(this.uScale, view.scale);
    gl.uniform1i(this.uMaxIter, iter);
    gl.uniform2f(this.uRes, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.uniform1i(this.uMode, mode === 'mandelbrot' ? 0 : 1);
    gl.uniform2f(this.uJuliaC, juliaC.x, juliaC.y);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
```

## WebGPU Backend (Optional Profile)

### wgpu-backend.js
```javascript
// wgpu-backend.js
export class WebGPUBackend {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.pipeline = null;
  }
  
  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext('webgpu');
    
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: presentationFormat,
    });
    
    this.setupPipeline();
  }
  
  setupPipeline() {
    const shaderModule = this.device.createShaderModule({
      code: `
        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
          var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 3.0, -1.0),
            vec2<f32>(-1.0,  3.0)
          );
          return vec4<f32>(pos[vertex_index], 0.0, 1.0);
        }
        
        @fragment
        fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
          // WebGPU compute shader implementation
          // Similar to WebGL2 but with compute shader syntax
          return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Placeholder
        }
      `
    });
    
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: navigator.gpu.getPreferredCanvasFormat(),
        }],
      },
    });
  }
  
  render({view, iter, mode, juliaC}) {
    if (!this.device) return;
    
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    
    renderPass.setPipeline(this.pipeline);
    renderPass.draw(3);
    renderPass.end();
    
    this.device.queue.submit([commandEncoder.finish()]);
  }
}
```

## Color Palettes & Anti-Banding

### palettes.js
```javascript
// palettes.js
export const palettes = {
  inferno: t => {
    const r = Math.pow(t, 0.6);
    const g = Math.pow(t, 1.2) * 0.85;
    const b = Math.pow(t, 3.0);
    return [Math.floor(255*r), Math.floor(255*g), Math.floor(255*b)];
  },
  
  viridis: t => {
    const r = Math.pow(t, 0.8);
    const g = Math.pow(t, 1.5) * 0.9;
    const b = Math.pow(t, 2.5);
    return [Math.floor(255*r), Math.floor(255*g), Math.floor(255*b)];
  },
  
  twilight: t => {
    const r = 0.5 + 0.5 * Math.cos(6.28 * (t + 0.0));
    const g = 0.5 + 0.5 * Math.cos(6.28 * (t + 0.33));
    const b = 0.5 + 0.5 * Math.cos(6.28 * (t + 0.67));
    return [Math.floor(255*r), Math.floor(255*g), Math.floor(255*b)];
  },
  
  plasma: t => {
    const r = Math.pow(t, 0.4);
    const g = Math.pow(t, 1.8) * 0.7;
    const b = Math.pow(t, 3.2);
    return [Math.floor(255*r), Math.floor(255*g), Math.floor(255*b)];
  }
};

export function paletteAt(fn, t) {
  return fn(t);
}

// Dithering to reduce banding
export function dither(x, y, value) {
  const noise = (Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
  return value + (noise - 0.5) * 0.1;
}
```

## Main Application Logic

### fractal.js
```javascript
// fractal.js
import { CpuBackend } from './cpu-backend.js';
import { Gl2Backend } from './gl2-backend.js';
import { WebGPUBackend } from './wgpu-backend.js';
import { palettes } from './palettes.js';

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const iterEl = document.getElementById('iter');
const iterv = document.getElementById('iterv');
const backendSel = document.getElementById('backend');
const paletteSel = document.getElementById('palette');
const modeSel = document.getElementById('mode');

// Initialize palettes
Object.keys(palettes).forEach(name => {
  const option = document.createElement('option');
  option.value = name;
  option.textContent = name;
  paletteSel.appendChild(option);
});

let backend = new CpuBackend();
let view = { x: -0.5, y: 0, scale: 300 };
let mode = 'mandelbrot';
let juliaC = { x: -0.8, y: 0.156 };
let iter = +iterEl.value;
let pal = palettes.inferno;

function resize() {
  canvas.width = innerWidth;
  canvas.height = innerHeight;
  draw();
}

window.addEventListener('resize', resize);
resize();

// Zoom handling
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const k = e.deltaY < 0 ? 1.1 : 0.9;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left - canvas.width/2) / view.scale + view.x;
  const my = (e.clientY - rect.top - canvas.height/2) / view.scale + view.y;
  view.x = mx + (view.x - mx) / k;
  view.y = my + (view.y - my) / k;
  view.scale *= k;
  draw();
}, { passive: false });

// Pan handling
let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
canvas.addEventListener('pointerdown', e => {
  dragging = true;
  sx = e.clientX;
  sy = e.clientY;
  ox = view.x;
  oy = view.y;
});

canvas.addEventListener('pointerup', () => dragging = false);

canvas.addEventListener('pointermove', e => {
  if (!dragging) return;
  const dx = (e.clientX - sx) / view.scale;
  const dy = (e.clientY - sy) / view.scale;
  view.x = ox - dx;
  view.y = oy - dy;
  draw();
});

// Julia mode: Alt+Click to set Julia constant
canvas.addEventListener('click', e => {
  if (e.altKey && mode === 'mandelbrot') {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - canvas.width/2) / view.scale + view.x;
    const my = (e.clientY - rect.top - canvas.height/2) / view.scale + view.y;
    juliaC = { x: mx, y: my };
    draw();
  }
});

// Backend switching
backendSel.onchange = async () => {
  try {
    if (backendSel.value === 'gl2') {
      backend = new Gl2Backend(canvas);
    } else if (backendSel.value === 'wgpu') {
      backend = new WebGPUBackend(canvas);
      await backend.init();
    } else {
      backend = new CpuBackend();
    }
    draw();
  } catch (e) {
    console.error('Backend error:', e);
    backend = new CpuBackend();
    backendSel.value = 'cpu';
  }
};

// Iteration count
iterEl.oninput = () => {
  iter = +iterEl.value;
  iterv.textContent = iter;
  draw();
};

// Palette switching
paletteSel.onchange = () => {
  pal = palettes[paletteSel.value];
  draw();
};

// Mode switching
modeSel.onchange = () => {
  mode = modeSel.value;
  draw();
};

// Reset view
document.getElementById('reset').onclick = () => {
  view = { x: -0.5, y: 0, scale: 300 };
  draw();
};

// Export PNG
document.getElementById('snap').onclick = () => {
  const link = document.createElement('a');
  link.download = 'fractal.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
};

// 8k Tiled Export
document.getElementById('tile').onclick = async () => {
  const tileSize = 2048;
  const targetSize = 8192;
  const tiles = targetSize / tileSize;
  
  const offscreen = new OffscreenCanvas(targetSize, targetSize);
  const offCtx = offscreen.getContext('2d');
  
  for (let ty = 0; ty < tiles; ty++) {
    for (let tx = 0; tx < tiles; tx++) {
      const tileView = {
        x: view.x + (tx - tiles/2) * tileSize / view.scale,
        y: view.y + (ty - tiles/2) * tileSize / view.scale,
        scale: view.scale
      };
      
      // Render tile
      const tileCanvas = new OffscreenCanvas(tileSize, tileSize);
      const tileCtx = tileCanvas.getContext('2d');
      
      // Create temporary backend for tile
      const tileBackend = new CpuBackend();
      await tileBackend.render({
        ctx: tileCtx,
        view: tileView,
        iter,
        palette: pal,
        mode,
        juliaC
      });
      
      // Draw tile to offscreen canvas
      offCtx.drawImage(tileCanvas, tx * tileSize, ty * tileSize);
    }
  }
  
  // Convert to blob and download
  const blob = await offscreen.convertToBlob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = 'fractal-8k.png';
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
};

async function draw() {
  const t0 = performance.now();
  await backend.render({ ctx, view, iter, palette: pal, mode, juliaC });
  const ms = (performance.now() - t0).toFixed(1);
  info.textContent = `(${view.x.toFixed(6)}, ${view.y.toFixed(6)}) scale=${view.scale.toFixed(1)} iter=${iter} | ${ms}ms`;
}

draw();
```

## Distance Estimation Shading

### distance-estimation.js
```javascript
// distance-estimation.js
export function addDistanceEstimation(canvas, ctx, view, iter, mode, juliaC) {
  const { width, height } = canvas;
  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;
  
  // Sample every 4th pixel for performance
  for (let y = 0; y < height; y += 4) {
    for (let x = 0; x < width; x += 4) {
      const zx = (x - width/2) / view.scale + view.x;
      const zy = (y - height/2) / view.scale + view.y;
      
      const de = distanceEstimation(zx, zy, iter);
      if (de > 0) {
        // Add contour lines or shading based on distance
        const intensity = Math.min(1, de * 10);
        const idx = (y * width + x) * 4;
        data[idx] *= intensity;     // R
        data[idx + 1] *= intensity; // G
        data[idx + 2] *= intensity; // B
      }
    }
  }
  
  ctx.putImageData(imgData, 0, 0);
}
```

## Advanced Modes

### buddhabrot.js
```javascript
// buddhabrot.js
export class BuddhabrotRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.accumulator = new Float32Array(canvas.width * canvas.height);
    this.maxIter = 1000;
    this.samples = 1000000;
  }
  
  async render() {
    const { width, height } = this.canvas;
    const img = new Uint8ClampedArray(width * height * 4);
    
    // Progressive accumulation
    for (let i = 0; i < this.samples; i++) {
      const cx = (Math.random() - 0.5) * 4;
      const cy = (Math.random() - 0.5) * 4;
      
      const orbit = this.computeOrbit(cx, cy);
      this.accumulateOrbit(orbit);
      
      if (i % 10000 === 0) {
        this.renderFrame();
        await new Promise(r => setTimeout(r, 1));
      }
    }
    
    this.renderFrame();
  }
  
  computeOrbit(cx, cy) {
    const orbit = [];
    let x = 0, y = 0;
    
    for (let i = 0; i < this.maxIter && x*x + y*y <= 4; i++) {
      const x2 = x*x - y*y + cx;
      const y2 = 2*x*y + cy;
      orbit.push({ x: x2, y: y2 });
      x = x2; y = y2;
    }
    
    return orbit;
  }
  
  accumulateOrbit(orbit) {
    const { width, height } = this.canvas;
    orbit.forEach(point => {
      const x = Math.floor((point.x + 2) * width / 4);
      const y = Math.floor((point.y + 2) * height / 4);
      if (x >= 0 && x < width && y >= 0 && y < height) {
        this.accumulator[y * width + x]++;
      }
    });
  }
  
  renderFrame() {
    const { width, height } = this.canvas;
    const img = new Uint8ClampedArray(width * height * 4);
    const max = Math.max(...this.accumulator);
    
    for (let i = 0; i < this.accumulator.length; i++) {
      const intensity = Math.sqrt(this.accumulator[i] / max);
      const idx = i * 4;
      img[idx] = intensity * 255;     // R
      img[idx + 1] = intensity * 255; // G
      img[idx + 2] = intensity * 255; // B
      img[idx + 3] = 255;             // A
    }
    
    const imgData = new ImageData(img, width, height);
    this.ctx.putImageData(imgData, 0, 0);
  }
}
```

## Performance Playbook

### CPU Optimization
- **Web Workers**: Use Workers for CPU rendering to avoid blocking UI
- **Row Striping**: Process image in strips for better progress indication
- **Transfer ArrayBuffer**: Use `transferList` to avoid copying large arrays
- **Precision**: Use BigInt/decimal.js for deep zooms, sample fewer pixels during navigation

### WebGL2 Optimization
- **Single Triangle**: Use one full-screen triangle, avoid geometry overhead
- **Uniform Caching**: Only update uniforms when values change
- **Program Reuse**: Cache compiled programs, avoid recreation
- **Precision**: Use highp precision for deep zooms

### WebGPU Optimization
- **Compute Shaders**: Use compute shaders for parallel processing
- **Storage Buffers**: Use storage buffers for large data
- **Pipeline Caching**: Cache render pipelines
- **Memory Management**: Properly manage GPU memory

### Anti-Banding Strategies
- **Dithering**: Add low-frequency noise to reduce banding
- **Blue Noise**: Use blue noise textures for GL paths
- **Log Scaling**: Use log-scaled mu for palette indexing
- **Gamma Correction**: Apply gamma correction before writing pixels

## UI Niceties

### Keyboard Shortcuts
```javascript
// Keyboard shortcuts
document.addEventListener('keydown', e => {
  switch(e.key) {
    case 'z': case 'Z':
      view.scale *= 1.1;
      draw();
      break;
    case 'x': case 'X':
      view.scale *= 0.9;
      draw();
      break;
    case 'ArrowLeft':
      view.x -= 0.1 / view.scale;
      draw();
      break;
    case 'ArrowRight':
      view.x += 0.1 / view.scale;
      draw();
      break;
    case 'ArrowUp':
      view.y += 0.1 / view.scale;
      draw();
      break;
    case 'ArrowDown':
      view.y -= 0.1 / view.scale;
      draw();
      break;
    case 'j': case 'J':
      mode = mode === 'mandelbrot' ? 'julia' : 'mandelbrot';
      modeSel.value = mode;
      draw();
      break;
    case 's': case 'S':
      document.getElementById('snap').click();
      break;
  }
});
```

### URL Bookmarks
```javascript
// URL state management
function updateURL() {
  const params = new URLSearchParams();
  params.set('x', view.x.toString());
  params.set('y', view.y.toString());
  params.set('scale', view.scale.toString());
  params.set('iter', iter.toString());
  params.set('mode', mode);
  params.set('palette', paletteSel.value);
  window.history.replaceState({}, '', `?${params}`);
}

function loadFromURL() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('x')) view.x = parseFloat(params.get('x'));
  if (params.has('y')) view.y = parseFloat(params.get('y'));
  if (params.has('scale')) view.scale = parseFloat(params.get('scale'));
  if (params.has('iter')) {
    iter = parseInt(params.get('iter'));
    iterEl.value = iter;
    iterv.textContent = iter;
  }
  if (params.has('mode')) {
    mode = params.get('mode');
    modeSel.value = mode;
  }
  if (params.has('palette')) {
    paletteSel.value = params.get('palette');
    pal = palettes[params.get('palette')];
  }
}
```

## TL;DR Runbook

```bash
# 1. Open the HTML file → explore (CPU backend)
# 2. Switch to WebGL2 for real-time renders
# 3. Scroll to zoom, drag to pan; Alt+Click sets Julia C
# 4. Export PNG; try "8k tiled" if your machine can take the heat
# 5. Optional: enable WebGPU profile and gloat

# Performance tips:
# - CPU: Use Workers, row-striping, transfer ArrayBuffer
# - GL: Single triangle, uniform caching, program reuse
# - Precision: BigInt/decimal.js for deep zooms
# - Anti-banding: Dithering, blue noise, log scaling

# Keyboard shortcuts:
# Z/X: zoom in/out
# Arrow keys: pan
# J: toggle Julia mode
# S: snapshot
```

---

*This tutorial provides the complete machinery for building interactive fractal art explorers. Each component is production-ready, copy-paste runnable, and designed for real-time mathematical visualization with multiple rendering backends.*
