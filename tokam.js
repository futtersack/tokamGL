'use strict';
/*---------------------------------------------------
/*
/*     Technical stuff declaration : Class, Vertex, Shaders
/*
/*-------------------------------------------------*/
const canvas = document.getElementsByTagName('canvas')[0];
resizeCanvas();
const { gl, ext } = getWebGLContext(canvas);

//Class definition
class Pointer{
  constructor () {
    this.id = -1;
    this.texcoordX = 0;
    this.texcoordY = 0;
    this.prevTexcoordX = 0;
    this.prevTexcoordY = 0;
    this.deltaX = 0;
    this.deltaY = 0;
    this.down = false;
    this.moved = false;
    }
}

class Material {
    constructor (vertexShader, fragmentShaderSource) {
        this.vertexShader = vertexShader;
        this.fragmentShaderSource = fragmentShaderSource;
        this.activeProgram = null;
        this.uniforms = [];
      
        let fragmentShader = compileShader(gl.FRAGMENT_SHADER, this.fragmentShaderSource);
        let program = createProgram(this.vertexShader, fragmentShader);

        this.uniforms = getUniforms(program);
        this.activeProgram = program;
    }

    bind () {
        gl.useProgram(this.activeProgram);
    }
}

class Program {
    constructor (vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = createProgram(vertexShader, fragmentShader);
        this.uniforms = getUniforms(this.program);
    }

    bind () {
        gl.useProgram(this.program);
    }
}

const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
    precision highp float;

    attribute vec2 aPosition;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;

    void main () {
        vUv = aPosition * 0.5 + 0.5;
        vL = vUv - vec2(texelSize.x, 0.0);
        vR = vUv + vec2(texelSize.x, 0.0);
        vT = vUv + vec2(0.0, texelSize.y);
        vB = vUv - vec2(0.0, texelSize.y);
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`);

const copyShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying highp vec2 vUv;
    uniform sampler2D uTexture;

    void main () {
        gl_FragColor = texture2D(uTexture, vUv);
    }
`);

const initializeShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying highp vec2 vUv;
    uniform vec4 value;
    uniform float perturb;
    #define M_PI 3.1415926535897932384626433832795
    
    void main () {
        float eps=0.;
        if(perturb==1.0){
          float p=vUv.x-0.2;
          eps+=1e-2*sin(2.0*M_PI*vUv.y*8.0)*exp(-(p*p)/(0.12*0.12));
          eps+=1e-3*cos(2.0*M_PI*vUv.y*13.0)*exp(-(p*p)/(0.12*0.12));
        }
        gl_FragColor = vec4(value.r+eps,value.g,value.b,value.a) ;
    }
`);


const displayShaderSource = `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;

    uniform sampler2D uTexture;
    uniform vec2 texelSize;
    uniform int numField;
    void main () {
        float value;
        vec3 color;
        if(numField==1) {
          value = texture2D(uTexture, vUv).r/8.;
          color.r=min(1.0,max(3.*(value-0.10),0.));
          color.g=min(1.0,max(3.*(value-0.37),0.));
          color.b=min(1.0,max(4.*(value-0.74),0.));
          }
        if(numField==2) {
          value = texture2D(uTexture, vUv).g*15.+0.5;
          color.r=min(1.,2.*exp(-((value-.7)/0.24)*((value-.7)/0.24)));
          color.g=min(1.,exp(-((value-.5)/0.25)*((value-.5)/0.25)));
          color.b=min(1.,2.*exp(-((value-.3)/0.24)*((value-.3)/0.24)));
          }
        if(numField==3) value = texture2D(uTexture, vUv).b/2.;
        if(numField==4) {
          value = (texture2D(uTexture, vUv).a-2.8388)/1.+0.5;
          color.r=min(1.,2.*exp(-((value-.7)/0.24)*((value-.7)/0.24)));
          color.g=min(1.,exp(-((value-.5)/0.25)*((value-.5)/0.25)));
          color.b=min(1.,2.*exp(-((value-.3)/0.24)*((value-.3)/0.24)));
          }
        gl_FragColor = vec4(color, 1.0);
    }
`;

const polarizeShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    
    uniform float aspectRatio;
    uniform vec2 texelSize;
    uniform vec2 point;
    uniform float Lp;
    uniform float Vp;
    uniform sampler2D uPhiWall;

    void main () {
        float px = vUv.x - point.x;
        float py = vUv.y - point.y;
        //p.x *= aspectRatio;
        float PhiW = min(texture2D(uPhiWall, vUv).r, Vp *
                   exp(-(py*py)/((Lp*texelSize.y)*(Lp*texelSize.y)))*
                   exp(-(px*px)/((Lp*texelSize.x)*(Lp*texelSize.x))));
        gl_FragColor = vec4(PhiW,0.,0.,0.);
        //gl_FragColor = vec4(texture2D(uFields, vUv).r,texture2D(uFields, vUv).gb,texture2D(uFields, vUv).a+PhiW);
    }
`);

const blobShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;
    uniform sampler2D uFields;
    uniform float aspectRatio;
    uniform vec2 move;
    uniform vec2 point;
    uniform vec4 attributes;
    // den,temp,size

    void main () {
        vec2 p = vUv - point.xy;
        p.x *= aspectRatio;
        float size=attributes.b;
        vec2 blob = attributes.rg*exp(-dot(p, p) / (size*texelSize.x*size*texelSize.y));
        vec4 base = texture2D(uFields, vUv).rgba;
        float Wx = 50.*move.x*(
                  sign(vT.y-point.y)*exp(-(vT.y-point.y)*(vT.y-point.y)/(4.*texelSize.y)/(4.*texelSize.y))
                 +sign(vB.y-point.y)*exp(-(vB.y-point.y)*(vB.y-point.y)/(4.*texelSize.y)/(4.*texelSize.y)))
                  *exp(-(vUv.x-point.x)*(vUv.x-point.x)/(texelSize.x)/(texelSize.x));
        float Wy = 50.*move.y*(
                  -sign(vL.x-point.x)*exp(-(vL.x-point.x)*(vL.x-point.x)/(4.*texelSize.x)/(4.*texelSize.x))
                  -sign(vR.x-point.x)*exp(-(vR.x-point.x)*(vR.x-point.x)/(4.*texelSize.x)/(4.*texelSize.x)))
                  *exp(-(vUv.y-point.y)*(vUv.y-point.y)/(texelSize.y)/(texelSize.y));
        gl_FragColor = vec4(base.r + blob.r,base.g + Wx + Wy ,base.ba);
    }
`);

const evolutionShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    
    uniform sampler2D uFields;
    uniform sampler2D uPhiWall;
    
    uniform vec2 texelSize;
    uniform float dt;
    uniform float lambda;
    uniform float sigma;
    uniform float g;
    uniform float S;
    uniform float D;
    uniform float nu;
    
    vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
        vec2 st = uv / tsize - 0.5;

        vec2 iuv = floor(st);
        vec2 fuv = fract(st);

        vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
        vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
        vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
        vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);

        return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
    }
    void main () {
    
        vec4 L = texture2D(uFields, vL).rgba;
        vec4 R = texture2D(uFields, vR).rgba;
        vec4 T = texture2D(uFields, vT).rgba;
        vec4 B = texture2D(uFields, vB).rgba;
        vec4 C = texture2D(uFields, vUv).rgba;
        
        float N = texture2D(uFields, vUv).r;
        float W = texture2D(uFields, vUv).g;
        float Phi = texture2D(uFields, vUv).a;
        
        float xs = vUv.x - 0.2 ;
        float source= S * exp( -(xs * xs) / (0.06*0.06));
        //float fb = exp(lambda-(Phi));
        float fb = exp(lambda-(Phi-texture2D(uPhiWall, vUv).r));
        float lossN = sigma*N*fb;
        float lossW = sigma*(1.0-fb);
        float curvature = g*(log(T.r)-log(B.r))*0.5;
        vec3 diffusion = (T.rgb-2.0*C.rgb+B.rgb + R.rgb-2.0*C.rgb+L.rgb);
        vec2 velocity = vec2(-T.a+B.a , R.a-L.a);
    #ifdef MANUAL_FILTERING
        vec2 coord = vUv - dt * velocity.xy * texelSize;
        
        N = bilerp(uFields, coord, texelSize).r + dt * ( source - lossN + D*diffusion.r);
        W = bilerp(uFields, coord, texelSize).g + dt * ( lossW - curvature + nu*diffusion.g );
    #else
        vec2 coord = vUv - dt * velocity * texelSize;
        N = texture2D(uFields, coord).r + dt * ( source - lossN + D*diffusion.r);
        W = texture2D(uFields, coord).g + dt * ( lossW - curvature + nu*diffusion.g );
    #endif
        
        gl_FragColor = vec4(N,W,1.0,Phi);
    }`
,ext.supportLinearFiltering ? null : '#define MANUAL_FILTERING');

const potentialShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uFields;

    void main () {
        float L = texture2D(uFields, vL).a;
        float R = texture2D(uFields, vR).a;
        float T = texture2D(uFields, vT).a;
        float B = texture2D(uFields, vB).a;
        float C = texture2D(uFields, vUv).a;
        float W = texture2D(uFields, vUv).g;
        float potential = (L + R + B + T - W) * 0.25;
        gl_FragColor = vec4(texture2D(uFields, vUv).rgb, potential);
    }
`);


const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    return (destination) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }
})();


/*---------------------------------------------------
/*
/*     Initialization
/*
/*-------------------------------------------------*/

const dt = 1.;
const copyProgram            = new Program(baseVertexShader, copyShader);
const initializeProgram      = new Program(baseVertexShader, initializeShader);
const polarizeProgram           = new Program(baseVertexShader, polarizeShader);
const blobProgram           = new Program(baseVertexShader, blobShader);
const evolutionProgram       = new Program(baseVertexShader, evolutionShader);
const potentialProgram       = new Program(baseVertexShader, potentialShader);
const displayMaterial = new Material(baseVertexShader, displayShaderSource);

let fields;
let phiWall;
let pointer = new Pointer();

let config = {
    ITERATION_RATE:0.0,
    FIELD:1,
    SIM_RESOLUTION: 512,
    LAMBDA:2.8388,
    SIGMA:1.6579e-4,
    G:5.72E-4,
    S:1e-2,
    D:1e-3,
    NU:1e-3,
    SOLVER_ITERATIONS: 20,
    BACK_COLOR: { r: 0, g: 0, b: 0 },
    LP: 8,
    VP: -3,
    BLOB_SIZE:4,
    BLOB_DEN:5e-1,
    PAUSED: false,
    ACTION:1,
}

function startGUI () {
    var gui = new dat.GUI({ width: 150 });
    gui.add(config, 'ITERATION_RATE', 200.).step(1e-2).name('ItPS').listen();
    gui.add(config, 'FIELD', { 'N': 1, 'Phi': 4, 'W': 2 }).name('Field');
    let parametersFolder = gui.addFolder('Parameters');
    parametersFolder.add(config, 'SIM_RESOLUTION', { '32': 32, '64': 64, '128': 128, '256': 256, '512': 512 }).name('sim resolution').onFinishChange(initFramebuffers);
    parametersFolder.add(config, 'LAMBDA').name('Lambda').min(0).max(4).step(1e-1);
    parametersFolder.add(config, 'SIGMA').name('Sigma').min(0).max(1e-3).step(1e-5);
    parametersFolder.add(config, 'G').name('g').min(0).max(1e-3).step(1e-5);
    parametersFolder.add(config, 'D').name('D').min(0).max(1e-1).step(0.001);
    parametersFolder.add(config, 'NU').name('Nu').min(0).max(1e-1).step(0.001);
    parametersFolder.add(config, 'S').name('S').min(0).max(1e-1).step(0.001);
    gui.add(config, 'ACTION', { 'Polarize': 1, 'Blobs': 2}).name('Action');
    let probes = gui.addFolder('Polarize');
    probes.add(config, 'LP').name('Size').min(8).max(32).step(1);
    probes.add(config, 'VP').name('Voltage').min(-10).max(10).step(0.25);
    let blobs = gui.addFolder('Blobs');
    blobs.add(config, 'BLOB_SIZE').name('Blob size').min(1).max(12).step(1);
    blobs.add(config, 'BLOB_DEN').name('Density').min(1e-2).max(1.).step(1e-2);
    gui.add(config, 'PAUSED').name('paused').listen();
    parametersFolder.open();
    var sliders=document.querySelectorAll(".slider");
    Array.prototype.filter.call(sliders, function(slider){
      slider.remove();
    });
    var sliders=document.querySelectorAll(".has-slider");
    Array.prototype.filter.call(sliders, function(slider){
      slider.classList.remove("has-slider");
    })
    return gui;
}

var gui=startGUI();
let lastUpdateTime;
let it=0;
initFramebuffers();
init();
update();

function init () {
    initializeProgram.bind();
    gl.uniform1f(initializeProgram.uniforms.perturb, 1.);
    gl.uniform4fv(initializeProgram.uniforms.value, [0.1,0.,1.,config.LAMBDA]);
    blit(fields.write.fbo);
    fields.swap();
    initializeProgram.bind();
    gl.uniform1f(initializeProgram.uniforms.perturb, 0.);
    gl.uniform4fv(initializeProgram.uniforms.value, [0.,0.,0.,0.]);
    blit(phiWall.read.fbo);
    lastUpdateTime = Date.now();
}

function update () {
    it++;
    let now = Date.now();
    if (resizeCanvas())
        initFramebuffers();
    applyInputs();

    if (!config.PAUSED){
        step(dt);
        if(it%10==0){
          let rate = 10./((now - lastUpdateTime) / 1000.);
          lastUpdateTime = now;
          config.ITERATION_RATE=rate;
          gui.__controllers[0].updateDisplay();
        }
    }
  
    requestAnimationFrame(update);
    render(null);

}


function getWebGLContext (canvas) {
    const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };

    let gl = canvas.getContext('webgl2', params);
    const isWebGL2 = !!gl;
    if (!isWebGL2)
        gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);

    let halfFloat;
    let supportLinearFiltering;
    if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float');
        supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
    } else {
        halfFloat = gl.getExtension('OES_texture_half_float');
        supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    let formatRGBA;
    let formatRG;
    let formatR;

    if (isWebGL2)
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
    }
    else
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    }

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
            supportLinearFiltering
        }
    };
}

function getSupportedFormat (gl, internalFormat, format, type)
{
    if (!supportRenderTextureFormat(gl, internalFormat, format, type))
    {
        switch (internalFormat)
        {
            case gl.R16F:
                return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
            case gl.RG16F:
                return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
            default:
                return null;
        }
    }

    return {
        internalFormat,
        format
    }
}

function supportRenderTextureFormat (gl, internalFormat, format, type) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);
  
    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    return status == gl.FRAMEBUFFER_COMPLETE;
}





function createProgram (vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw gl.getProgramInfoLog(program);

    return program;
}

function getUniforms (program) {
    let uniforms = [];
    let uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
        let uniformName = gl.getActiveUniform(program, i).name;
        uniforms[uniformName] = gl.getUniformLocation(program, uniformName);
    }
    return uniforms;
}

function compileShader (type, source, define) {
    return compileShader (type, define+'\n'+source);
};

function compileShader (type, source) {

    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw gl.getShaderInfoLog(shader);

    return shader;
};


function initFramebuffers () {
    let simRes = getResolution(config.SIM_RESOLUTION);

    const texType = ext.halfFloatTexType;
    const rgba    = ext.formatRGBA;
    const rg      = ext.formatRG;
    const r       = ext.formatR;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    if (fields == null){
        fields = createDoubleFBO(simRes.width, simRes.height, rgba.internalFormat, rgba.format, texType, filtering);
        phiWall = createDoubleFBO(simRes.width, simRes.height, rgba.internalFormat, rgba.format, texType, filtering);
        }
    else{
        fields = resizeDoubleFBO(fields, simRes.width, simRes.height, rgba.internalFormat, rgba.format, texType, filtering);
        phiWall = resizeDoubleFBO(phiWall, simRes.width, simRes.height, rgba.internalFormat, rgba.format, texType, filtering);
        }
    //phiWall = createFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, filtering);
}

function createFBO (w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    let texelSizeX = 1.0 / w;
    let texelSizeY = 1.0 / h;

    return {
        texture,
        fbo,
        width: w,
        height: h,
        texelSizeX,
        texelSizeY,
        attach (id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };
}

function createDoubleFBO (w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);

    return {
        width: w,
        height: h,
        texelSizeX: fbo1.texelSizeX,
        texelSizeY: fbo1.texelSizeY,
        get read () {
            return fbo1;
        },
        set read (value) {
            fbo1 = value;
        },
        get write () {
            return fbo2;
        },
        set write (value) {
            fbo2 = value;
        },
        swap () {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    }
}

function resizeFBO (target, w, h, internalFormat, format, type, param) {
    let newFBO = createFBO(w, h, internalFormat, format, type, param);
    copyProgram.bind();
    gl.uniform1i(copyProgram.uniforms.uTexture, target.attach(0));
    blit(newFBO.fbo);
    return newFBO;
}

function resizeDoubleFBO (target, w, h, internalFormat, format, type, param) {
    if (target.width == w && target.height == h)
        return target;
    target.read = resizeFBO(target.read, w, h, internalFormat, format, type, param);
    target.write = createFBO(w, h, internalFormat, format, type, param);
    target.width = w;
    target.height = h;
    target.texelSizeX = 1.0 / w;
    target.texelSizeY = 1.0 / h;
    return target;
}


function step (dt) {
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, fields.width, fields.height);

    evolutionProgram.bind();
    gl.uniform2f(evolutionProgram.uniforms.texelSize, fields.texelSizeX, fields.texelSizeY);
    gl.uniform1i(evolutionProgram.uniforms.uFields, fields.read.attach(0));
    gl.uniform1i(evolutionProgram.uniforms.uPhiWall, phiWall.read.attach(1));
    gl.uniform1f(evolutionProgram.uniforms.dt, dt);
    gl.uniform1f(evolutionProgram.uniforms.lambda, config.LAMBDA);
    gl.uniform1f(evolutionProgram.uniforms.sigma, config.SIGMA);
    gl.uniform1f(evolutionProgram.uniforms.g, config.G);
    gl.uniform1f(evolutionProgram.uniforms.S, config.S);
    gl.uniform1f(evolutionProgram.uniforms.D, config.D);
    gl.uniform1f(evolutionProgram.uniforms.nu, config.NU);
    blit(fields.write.fbo);
    fields.swap();
  
    //clearProgram.bind();
    //gl.uniform1f(clearProgram.uniforms.lambda, config.LAMBDA);
    //gl.uniform1i(clearProgram.uniforms.uFields, fields.read.attach(0));
    //blit(fields.write.fbo);
    //fields.swap();
  
    potentialProgram.bind();
    gl.uniform2f(potentialProgram.uniforms.texelSize, fields.texelSizeX, fields.texelSizeY);
    for (let i = 0; i < config.SOLVER_ITERATIONS; i++) {
        gl.uniform1i(potentialProgram.uniforms.uFields, fields.read.attach(1));
        blit(fields.write.fbo);
        fields.swap();
    }

}

function polarize(x, y) {
    gl.viewport(0, 0, fields.width, fields.height);
    polarizeProgram.bind();

    gl.uniform1f(polarizeProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(polarizeProgram.uniforms.texelSize, fields.texelSizeX, fields.texelSizeY);
    gl.uniform2f(polarizeProgram.uniforms.point, x, y);
    gl.uniform1f(polarizeProgram.uniforms.Lp, config.LP);
    gl.uniform1f(polarizeProgram.uniforms.Vp, config.VP);
    gl.uniform1i(polarizeProgram.uniforms.uFields, phiWall.read.attach(0));
    blit(phiWall.write.fbo);
    //blit(fields.write.fbo);
    phiWall.swap();
}
function cancelPolarization(x, y) {
    gl.viewport(0, 0, fields.width, fields.height);
    initializeProgram.bind();
    gl.uniform1f(initializeProgram.uniforms.perturb, 0.);
    gl.uniform4fv(initializeProgram.uniforms.value, [0.,0.,0.,0.]);
    blit(phiWall.read.fbo);
    //phiWall.swap();
}

function blob(x, y) {
    let dx = pointer.deltaX;
    let dy = pointer.deltaY;
    gl.viewport(0, 0, fields.width, fields.height);
    blobProgram.bind();
    gl.uniform1i(blobProgram.uniforms.uFields, fields.read.attach(0));
    gl.uniform2f(blobProgram.uniforms.texelSize, fields.texelSizeX, fields.texelSizeY);
    gl.uniform1f(blobProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(blobProgram.uniforms.point, x, y);
    gl.uniform2f(blobProgram.uniforms.move, dx, dy);
    gl.uniform4fv(blobProgram.uniforms.attributes, [config.BLOB_DEN,0.0,config.BLOB_SIZE,0.0]);
    blit(fields.write.fbo);
    fields.swap();
}

function resizeCanvas () {
    let width = scaleByPixelRatio(canvas.clientWidth);
    let height = scaleByPixelRatio(canvas.clientHeight);
    if (canvas.width != width || canvas.height != height) {
        canvas.width = width;
        canvas.height = height;
        return true;
    }
    return false;
}



function render (target) {
    let width = target == null ? gl.drawingBufferWidth : target.width;
    let height = target == null ? gl.drawingBufferHeight : target.height;
    gl.viewport(0, 0, width, height);

    let fbo = target == null ? null : target.fbo;

    drawDisplay(fbo, width, height);
}

function drawDisplay (fbo, width, height) {
    displayMaterial.bind();
    gl.uniform1i(displayMaterial.uniforms.numField, config.FIELD);
    gl.uniform1i(displayMaterial.uniforms.uTexture, fields.read.attach(0));
    //gl.uniform1i(displayMaterial.uniforms.uTexture, phiWall.read.attach(0));
    blit(fbo);
}

function applyInputs () {
    if (config.ACTION==1) {
      if (pointer.down) polarize(pointer.texcoordX, pointer.texcoordY);
      else cancelPolarization();
    }
    if (config.ACTION==2) {
      if (pointer.down) blob(pointer.texcoordX, pointer.texcoordY);
    }
}

canvas.addEventListener('mousedown', e => {
    let posX = scaleByPixelRatio(e.offsetX);
    let posY = scaleByPixelRatio(e.offsetY);
    if (pointer == null)
        pointer = new pointerPrototype();
    updatePointerDownData(pointer, posX, posY);
});

canvas.addEventListener('mousemove', e => {
    if (!pointer.down) return;
    let posX = scaleByPixelRatio(e.offsetX);
    let posY = scaleByPixelRatio(e.offsetY);
    updatePointerMoveData(pointer, posX, posY);
});

window.addEventListener('mouseup', () => {
    updatePointerUpData(pointer);
});

canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        if(touches[i].identifier!=0) continue;
        let posX = scaleByPixelRatio(touches[i].pageX);
        let posY = scaleByPixelRatio(touches[i].pageY);
        updatePointerDownData(pointer, posX, posY);
    }
});

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        if(touches[i].identifier!=0) continue;
        if (!pointer.down) continue;
        let posX = scaleByPixelRatio(touches[i].pageX);
        let posY = scaleByPixelRatio(touches[i].pageY);
        updatePointerMoveData(pointer, posX, posY);
    }
}, false);

window.addEventListener('touchend', e => {
    const touches = e.changedTouches;
    for (let i = 0; i < touches.length; i++)
    {
        if(touches[i].identifier!=0) continue;
        
        if (pointer == null) continue;
        updatePointerUpData(pointer);
    }
});

window.addEventListener('keydown', e => {
    if (e.code === 'KeyP' || e.key === ' ')
        config.PAUSED = !config.PAUSED;
    if (e.code === 'KeyS')
        update();
});

function updatePointerDownData (pointer, posX, posY) {
    pointer.down = true;
    pointer.texcoordX = posX / canvas.width;
    pointer.texcoordY = 1.0 - posY / canvas.height;
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.deltaX = 0;
    pointer.deltaY = 0;
}

function updatePointerMoveData (pointer, posX, posY) {
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.texcoordX = posX / canvas.width;
    pointer.texcoordY = 1.0 - posY / canvas.height;
    pointer.deltaX = correctDeltaX(pointer.texcoordX - pointer.prevTexcoordX);
    pointer.deltaY = correctDeltaY(pointer.texcoordY - pointer.prevTexcoordY);
    pointer.moved = Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0;
}

function updatePointerUpData (pointer) {
    pointer.down = false;
}
function correctDeltaX (delta) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio < 1) delta *= aspectRatio;
    return delta;
}

function correctDeltaY (delta) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio > 1) delta /= aspectRatio;
    return delta;
}

function getResolution (resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1)
        aspectRatio = 1.0 / aspectRatio;

    let min = Math.round(resolution);
    let max = Math.round(resolution * aspectRatio);

    if (aspectRatio > 1)
        return { width: max, height: min };
    else
        return { width: min, height: max };
}

function getTextureScale (texture, width, height) {
    return {
        x: width / texture.width,
        y: height / texture.height
    };
}

function scaleByPixelRatio (input) {
    let pixelRatio = window.devicePixelRatio || 1;
    return Math.floor(input * pixelRatio);
}
