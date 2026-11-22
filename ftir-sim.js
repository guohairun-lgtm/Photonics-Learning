// FTIR Simulator Engine — unified tab switching + hidden-tab pause + physics + canvas rendering
(function(){
'use strict';

var isSimActive = false;
var buttons = document.querySelectorAll('.tab-switch button');
var panels = document.querySelectorAll('.tab-panel');
buttons.forEach(function(btn){
  btn.addEventListener('click', function(){
    buttons.forEach(function(b){ b.classList.remove('active'); });
    panels.forEach(function(p){ p.classList.remove('active'); });
    btn.classList.add('active');
    var target = btn.dataset.panel;
    document.getElementById('panel-' + target).classList.add('active');
    isSimActive = (target === 'simulation');
    if (isSimActive) setTimeout(renderAll, 30);
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
  });
});

var N=4096, HALF_N=N>>1, DT=0.5, DF=1000/(N*DT);
var C_UM_PER_FS=0.299792458, F_DISPLAY_MAX=220, DUAL_SPLIT=15, H_OVER_KB_THZ=47.99243;
var tAxis=new Float64Array(N); for(var i=0;i<N;i++) tAxis[i]=(i-HALF_N)*DT;
var E_src=new Float64Array(N), fftRe=new Float64Array(N), fftIm=new Float64Array(N);
var S_true=new Float64Array(HALF_N), S_ftir=new Float64Array(HALF_N);
var I_ac_raw=new Float64Array(N), I_norm=new Float64Array(N);
var state={shape:'gaussian',tauP:30,f0:100,tempK:1500,avgCount:0,rngSeed:202605,tauMax:150,tau:0,winType:'boxcar',isScanning:false,scanDir:1,animTime:0};
var metrics={r2:1.0,rmse:0.0,df:3.33};

function isIncoherent(s){ return s==='spontaneous'||s==='blackbody'; }
function createPRNG(seed){ var a=seed>>>0; return function(){ a|=0; a=(a+0x6D2B79F5)|0; var t=Math.imul(a^(a>>>15),1|a); t=(t+Math.imul(t^(t>>>7),61|t))^t; return ((t^(t>>>14))>>>0)/4294967296; }; }

function fft(re,im,inverse){
  var n=re.length, j=0;
  for(var i=0;i<n;i++){ if(i<j){ var tr=re[i]; re[i]=re[j]; re[j]=tr; var ti=im[i]; im[i]=im[j]; im[j]=ti; } var m=n>>1; while(m>=1&&j>=m){ j-=m; m>>=1; } j+=m; }
  var sign=inverse?1:-1;
  for(var len=2;len<=n;len<<=1){
    var halfLen=len>>1, angle=(sign*2*Math.PI)/len, wStepRe=Math.cos(angle), wStepIm=Math.sin(angle);
    for(var i0=0;i0<n;i0+=len){
      var wRe=1.0, wIm=0.0;
      for(var k=0;k<halfLen;k++){
        var uIdx=i0+k, vIdx=uIdx+halfLen;
        var vRe=re[vIdx]*wRe-im[vIdx]*wIm, vIm=re[vIdx]*wIm+im[vIdx]*wRe;
        re[vIdx]=re[uIdx]-vRe; im[vIdx]=im[uIdx]-vIm; re[uIdx]+=vRe; im[uIdx]+=vIm;
        var nextWRe=wRe*wStepRe-wIm*wStepIm; wIm=wRe*wStepIm+wIm*wStepRe; wRe=nextWRe;
      }
    }
  }
  if(inverse){ var invN=1.0/n; for(var i2=0;i2<n;i2++){ re[i2]*=invN; im[i2]*=invN; } }
}

function evalPulseField(t,shape,tauP,f0){
  var ps=2*Math.PI*1e-3, u=t/tauP;
  switch(shape){
    case 'gaussian': return Math.exp(-2*Math.LN2*u*u)*Math.cos(ps*f0*t);
    case 'sech': { var a=1.762747174*u; return (Math.abs(a)>30?0:1/Math.cosh(a))*Math.cos(ps*f0*t); }
    case 'lorentzian': return (1/(1+4*u*u))*Math.cos(ps*f0*t);
    case 'square': return (Math.abs(t)<=0.5*tauP?1:0)*Math.cos(ps*f0*t);
    case 'dual': { var env=Math.exp(-2*Math.LN2*u*u); return 0.5*env*(Math.cos(ps*(f0-DUAL_SPLIT)*t)+Math.cos(ps*(f0+DUAL_SPLIT)*t)); }
    default: return 0;
  }
}

function evalWindow(tau,tauMax,winType){
  var a=Math.abs(tau); if(a>tauMax) return 0; var r=a/tauMax;
  if(winType==='hann') return 0.5*(1+Math.cos(Math.PI*r));
  if(winType==='triangular') return 1-r;
  return 1.0;
}

function recomputePhysics(){
  var shape=state.shape, tauP=state.tauP, f0=state.f0, tempK=state.tempK, avgCount=state.avgCount, rngSeed=state.rngSeed, tauMax=state.tauMax, winType=state.winType;
  var psdRaw=new Float64Array(N), maxStrue=1e-18;
  if(!isIncoherent(shape)){
    for(var i=0;i<N;i++) E_src[i]=evalPulseField(tAxis[i],shape,tauP,f0);
    for(var i2=0;i2<N;i2++){ fftRe[i2]=E_src[(i2+HALF_N)&(N-1)]; fftIm[i2]=0; }
    fft(fftRe,fftIm,false);
    for(var k=0;k<N;k++){ var p=fftRe[k]*fftRe[k]+fftIm[k]*fftIm[k]; psdRaw[k]=p; if(k<HALF_N&&p>maxStrue) maxStrue=p; }
    for(var k2=0;k2<HALF_N;k2++) S_true[k2]=psdRaw[k2]/maxStrue;
  } else {
    var rand=createPRNG(rngSeed), targetPSD=new Float64Array(HALF_N);
    if(shape==='blackbody'){ for(var k3=1;k3<HALF_N;k3++){ var f=k3*DF, x=(H_OVER_KB_THZ*f)/tempK, val=x>650?0:(f*f*f)/(Math.exp(x)-1); targetPSD[k3]=val; if(val>maxStrue) maxStrue=val; } }
    else { var dfLor=1000/(Math.PI*tauP); for(var k4=1;k4<HALF_N;k4++){ var f2=k4*DF, u2=(2*(f2-f0))/dfLor, val2=1/(1+u2*u2); targetPSD[k4]=val2; if(val2>maxStrue) maxStrue=val2; } }
    for(var k5=0;k5<HALF_N;k5++) S_true[k5]=targetPSD[k5]/maxStrue;
    fftRe.fill(0); fftIm.fill(0);
    for(var k6=1;k6<HALF_N;k6++){ var u1=Math.max(1e-12,rand()), u2r=rand(), r=Math.sqrt(-2*Math.log(u1)), amp=Math.sqrt(S_true[k6]); fftRe[k6]=amp*r*Math.cos(2*Math.PI*u2r); fftIm[k6]=amp*r*Math.sin(2*Math.PI*u2r); fftRe[N-k6]=fftRe[k6]; fftIm[N-k6]=-fftIm[k6]; }
    fft(fftRe,fftIm,true);
    var sumSq=0; for(var i3=0;i3<N;i3++) sumSq+=fftRe[i3]*fftRe[i3];
    var normScale=0.45/(Math.sqrt(sumSq/N)||1);
    for(var i4=0;i4<N;i4++) E_src[i4]=fftRe[(i4+HALF_N)&(N-1)]*normScale;
    psdRaw.fill(0);
    for(var k7=1;k7<HALF_N;k7++){ var pVal=S_true[k7]; if(avgCount>0){ var expSum=0; for(var m=0;m<avgCount;m++) expSum+=-Math.log(Math.max(1e-12,rand())); pVal=S_true[k7]*(expSum/avgCount); } psdRaw[k7]=pVal; psdRaw[N-k7]=pVal; }
    maxStrue=1.0;
  }
  for(var i5=0;i5<N;i5++){ fftRe[i5]=psdRaw[i5]; fftIm[i5]=0; }
  fft(fftRe,fftIm,true);
  var eEnergy=Math.max(fftRe[0],1e-18);
  for(var i6=0;i6<N;i6++){ var rEE=fftRe[(i6+HALF_N)&(N-1)]; I_ac_raw[i6]=2*rEE; I_norm[i6]=1.0+rEE/eEnergy; }
  for(var i7=0;i7<N;i7++){ var si=(i7+HALF_N)&(N-1); fftRe[i7]=I_ac_raw[si]*evalWindow(tAxis[si],tauMax,winType); fftIm[i7]=0; }
  fft(fftRe,fftIm,false);
  for(var k8=0;k8<HALF_N;k8++) S_ftir[k8]=(0.5*fftRe[k8])/maxStrue;
  var kMax=Math.min(HALF_N-1,Math.floor(F_DISPLAY_MAX/DF)), meanTrue=0;
  for(var k9=0;k9<=kMax;k9++) meanTrue+=S_true[k9]; meanTrue/=(kMax+1);
  var ssRes=0, ssTot=0;
  for(var k10=0;k10<=kMax;k10++){ var diff=S_ftir[k10]-S_true[k10]; ssRes+=diff*diff; var dtot=S_true[k10]-meanTrue; ssTot+=dtot*dtot; }
  metrics={r2:ssTot>1e-12?Math.max(0,1-ssRes/ssTot):1, rmse:Math.sqrt(ssRes/(kMax+1)), df:1000/(2*tauMax)};
  updateMetricsUI();
}

function $(id){ return document.getElementById(id); }
var els={
  ctrlTauP:$('ctrl-taup'), ctrlF0:$('ctrl-f0'), ctrlTemp:$('ctrl-temp'), ctrlIncoherent:$('ctrl-incoherent'),
  labelTauP:$('label-taup'), hintTauPLeft:$('hint-taup-left'), hintTauPRight:$('hint-taup-right'),
  sliderTauP:$('slider-taup'), sliderF0:$('slider-f0'), sliderTemp:$('slider-temp'),
  sliderTauMax:$('slider-taumax'), sliderTau:$('slider-tau'),
  valTauP:$('val-taup'), valF0:$('val-f0'), valTemp:$('val-temp'), valTauMax:$('val-taumax'), valTau:$('val-tau'),
  labelShapeFormula:$('label-shape-formula'), btnReseed:$('btn-reseed'), btnScan:$('btn-scan'), textScan:$('text-scan'),
  btnZeroTau:$('btn-zero-tau'), btnReset:$('btn-reset'), btnPresetRes:$('btn-preset-res'),
  metricTau:$('metric-tau'), metricOpd:$('metric-opd'), metricDf:$('metric-df'), metricWavenum:$('metric-wavenum'),
  metricR2:$('metric-r2'), metricR2Badge:$('metric-r2-badge'), metricRmse:$('metric-rmse'),
  badgeOpdA:$('badge-opd-a'), tipBox:$('physics-tip-box'), tipText:$('physics-tip-text')
};
var canvases={ interferometer:$('canvas-interferometer'), timedomain:$('canvas-timedomain'), interferogram:$('canvas-interferogram'), spectrum:$('canvas-spectrum') };

function setupHiDPI(c){
  var rect=c.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2.5);
  var w=Math.max(10,Math.round(rect.width)), h=Math.max(10,Math.round(rect.height));
  if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){ c.width=Math.round(w*dpr); c.height=Math.round(h*dpr); }
  var ctx=c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx:ctx,w:w,h:h};
}

function updateMetricsUI(){
  var opdUm=state.tau*C_UM_PER_FS, wavenumRes=1e4/(2*state.tauMax*C_UM_PER_FS), r2Pct=metrics.r2*100;
  els.metricTau.textContent=(state.tau>=0?'+':'')+state.tau.toFixed(1)+' fs';
  els.metricOpd.textContent=(opdUm>=0?'+':'')+opdUm.toFixed(2)+' μm';
  els.badgeOpdA.textContent='OPD = '+(opdUm>=0?'+':'')+opdUm.toFixed(2)+' μm';
  els.metricDf.textContent=metrics.df.toFixed(2)+' THz';
  els.metricWavenum.textContent='('+wavenumRes.toFixed(0)+' cm⁻¹)';
  els.metricR2.textContent=r2Pct.toFixed(2)+'%';
  els.metricRmse.textContent=metrics.rmse.toFixed(4);
  els.metricR2Badge.textContent=r2Pct>=99?'极高吻合':(r2Pct>=92?'轻度展宽':'截断失真');
  els.tipText.textContent=r2Pct>=99?'最大扫描延时充分覆盖相干时间，FTIR 反演光谱与真实功率谱高度重合。':'当前分辨率 Δf = '+metrics.df.toFixed(2)+' THz，增大 τ_max 或启用切趾窗可改善光谱还原度。';
  els.tipBox.className='ftir-tip '+(r2Pct>=99?'ok':(r2Pct>=92?'info':'warn'));
}

function drawInterferometer(){
  var r=setupHiDPI(canvases.interferometer), ctx=r.ctx, w=r.w, h=r.h;
  if(w<=10) return;
  ctx.clearRect(0,0,w,h); ctx.fillStyle='#faf9f7'; ctx.fillRect(0,0,w,h);
  var ci=Math.max(0,Math.min(N-1,Math.round(state.tau/DT)+HALF_N)), currI=I_norm[ci];
  var cx=w*0.46, cy=h*0.50, srcX=Math.max(40,w*0.13), m1Y=Math.max(32,h*0.16), pdY=Math.min(h-26,h*0.85);
  var m2BaseX=Math.min(w-48,w*0.78), maxShiftPx=Math.min(32,Math.max(14,(w-m2BaseX)*0.65));
  var m2X=m2BaseX+(state.tau/Math.max(1,state.tauMax))*maxShiftPx;
  function db(x1,y1,x2,y2,c,lw){ ctx.save(); ctx.strokeStyle=c; ctx.lineWidth=lw; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); ctx.restore(); }
  var PURPLE='#7F77DD', sbc=isIncoherent(state.shape)?'rgba(239,159,39,0.55)':'rgba(127,119,221,0.55)';
  db(srcX+20,cy,cx,cy,sbc,3);
  db(cx,cy,cx,m1Y+6,'rgba(55,138,221,0.5)',2.5);
  db(cx,cy,m2X-6,cy,'rgba(239,159,39,0.5)',2.5);
  db(cx,cy,cx,pdY-12,'rgba(29,158,117,'+Math.min(1,0.25+0.37*currI)+')',2.2+currI*1.4);

  // Animated glowing wave-packet / photon dots
  var tc=(state.animTime*0.65)%1;
  function dp(x,y,c,rad){ ctx.save(); ctx.fillStyle=c; ctx.shadowColor=c; ctx.shadowBlur=8; ctx.beginPath(); ctx.arc(x,y,rad,0,Math.PI*2); ctx.fill(); ctx.restore(); }
  if(isIncoherent(state.shape)){
    for(var s=0;s<4;s++){
      var ph=(tc+s*0.25)%1;
      dp(srcX+22+(cx-srcX-22)*ph,cy,'#EF9F27',3);
      var tr=ph<0.5?ph*2:(1-ph)*2;
      dp(cx,cy-(cy-(m1Y+8))*tr,'#378ADD',2.8);
      dp(cx+(m2X-8-cx)*tr,cy,'#EF9F27',2.8);
      dp(cx,cy+(pdY-14-cy)*ph,'#1D9E75',2.6+currI*1.1);
    }
  } else if(tc<0.32){
    var p=tc/0.32; dp(srcX+22+(cx-srcX-22)*p,cy,PURPLE,4.5);
  } else if(tc<0.72){
    var p2=(tc-0.32)/0.40, tr2=p2<0.5?p2*2:(1-p2)*2;
    dp(cx,cy-(cy-(m1Y+8))*tr2,'#378ADD',3.8);
    dp(cx+(m2X-8-cx)*tr2,cy,'#EF9F27',3.8);
  } else {
    var p3=(tc-0.72)/0.28; dp(cx,cy+(pdY-14-cy)*p3,'#1D9E75',3.5+currI*1.4);
  }

  // Light source box
  ctx.save(); ctx.fillStyle='#ffffff'; ctx.strokeStyle=isIncoherent(state.shape)?'#EF9F27':'#7F77DD'; ctx.lineWidth=1.5;
  ctx.beginPath(); if(ctx.roundRect) ctx.roundRect(srcX-24,cy-16,44,32,6); else ctx.rect(srcX-24,cy-16,44,32); ctx.fill(); ctx.stroke();
  ctx.fillStyle=isIncoherent(state.shape)?'#854F0B':'#534AB7'; ctx.font='bold 10px -apple-system,sans-serif'; ctx.textAlign='center';
  ctx.fillText(state.shape==='blackbody'?'黑体源':(state.shape==='spontaneous'?'自发源':'光源'),srcX-2,cy+4); ctx.restore();

  // Beam splitter (45 deg)
  ctx.save(); ctx.translate(cx,cy); ctx.rotate(-Math.PI/4);
  ctx.fillStyle='rgba(127,119,221,0.12)'; ctx.strokeStyle='#5f5e5a'; ctx.lineWidth=1.5;
  ctx.fillRect(-20,-3,40,6); ctx.strokeRect(-20,-3,40,6); ctx.restore();
  ctx.fillStyle='#5f5e5a'; ctx.font='9.5px "SF Mono",monospace'; ctx.textAlign='right'; ctx.fillText('BS (50:50)',cx-16,cy+25);

  // Fixed mirror M1
  ctx.save(); ctx.fillStyle='#378ADD'; ctx.fillRect(cx-24,m1Y-4,48,6); ctx.fillStyle='#B5D4F4'; ctx.fillRect(cx-26,m1Y-9,52,5);
  ctx.fillStyle='#185FA5'; ctx.font='10.5px -apple-system,sans-serif'; ctx.textAlign='center'; ctx.fillText('定镜 M1',cx,m1Y-13); ctx.restore();

  // Moving mirror M2 + ZPD rail
  ctx.save(); ctx.strokeStyle='#e0ddd3'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(m2BaseX-maxShiftPx-6,cy+30); ctx.lineTo(m2BaseX+maxShiftPx+14,cy+30); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle='rgba(239,159,39,0.5)'; ctx.beginPath(); ctx.moveTo(m2BaseX,cy+24); ctx.lineTo(m2BaseX,cy+36); ctx.stroke();
  ctx.fillStyle='#888780'; ctx.font='9px "SF Mono",monospace'; ctx.textAlign='center'; ctx.fillText('ZPD',m2BaseX,cy+46);
  ctx.fillStyle='#EF9F27'; ctx.fillRect(m2X-4,cy-22,6,44); ctx.fillStyle='#FAC775'; ctx.fillRect(m2X+2,cy-24,6,48);
  var dxUm=0.5*state.tau*C_UM_PER_FS;
  ctx.fillStyle='#854F0B'; ctx.font='10.5px -apple-system,sans-serif'; ctx.textAlign='center'; ctx.fillText('动镜 M2',m2X+2,cy-30);
  ctx.font='9.5px "SF Mono",monospace'; ctx.fillStyle='#EF9F27'; ctx.fillText('Δx='+(dxUm>=0?'+':'')+dxUm.toFixed(1)+'μm',m2X+2,cy-43); ctx.restore();

  // Photodetector PD with radial glow
  ctx.save(); var gr=9+currI*8, grad=ctx.createRadialGradient(cx,pdY,2,cx,pdY,gr+8);
  grad.addColorStop(0,'rgba(29,158,117,'+Math.min(0.95,0.3+0.35*currI)+')'); grad.addColorStop(1,'rgba(29,158,117,0)');
  ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(cx,pdY,gr+8,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#ffffff'; ctx.strokeStyle='#1D9E75'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(cx,pdY,12,0,Math.PI,false); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.fillStyle='#0F6E56'; ctx.font='10.5px -apple-system,sans-serif'; ctx.textAlign='left'; ctx.fillText('探测器 PD',cx+18,pdY+2);
  ctx.font='9.5px "SF Mono",monospace'; ctx.fillStyle='#1D9E75'; ctx.fillText('I/I_DC='+currI.toFixed(2),cx+18,pdY+15); ctx.restore();
}

function drawAxes(ctx,w,h,pad,xMin,xMax,yMin,yMax,xLabel,yLabel){
  var plotW=w-pad.l-pad.r, plotH=h-pad.t-pad.b;
  var mapX=function(x){ return pad.l+((x-xMin)/(xMax-xMin))*plotW; };
  var mapY=function(y){ return pad.t+plotH-((y-yMin)/(yMax-yMin))*plotH; };
  ctx.strokeStyle='rgba(224,221,211,0.7)'; ctx.lineWidth=0.5; ctx.font='10px monospace'; ctx.fillStyle='#888780';
  ctx.textAlign='center'; for(var i=0;i<=6;i++){ var xv=xMin+(i/6)*(xMax-xMin), px=mapX(xv); ctx.beginPath(); ctx.moveTo(px,pad.t); ctx.lineTo(px,pad.t+plotH); ctx.stroke(); ctx.fillText(Math.round(xv),px,h-pad.b+13); }
  ctx.textAlign='right'; for(var i2=0;i2<=4;i2++){ var yv=yMin+(i2/4)*(yMax-yMin), py=mapY(yv); ctx.beginPath(); ctx.moveTo(pad.l,py); ctx.lineTo(pad.l+plotW,py); ctx.stroke(); ctx.fillText(yv.toFixed(1),pad.l-5,py+3); }
  ctx.fillStyle='#5f5e5a'; ctx.font='10px -apple-system,sans-serif'; ctx.textAlign='center'; ctx.fillText(xLabel,pad.l+plotW/2,h-4);
  return {mapX:mapX,mapY:mapY,plotW:plotW,plotH:plotH};
}

function sampleFieldAt(t){
  if(!isIncoherent(state.shape)) return evalPulseField(t,state.shape,state.tauP,state.f0);
  var idxF=((t/DT+HALF_N)%N+N)%N, i0=Math.floor(idxF), i1=(i0+1)&(N-1), frac=idxF-i0;
  return E_src[i0]*(1-frac)+E_src[i1]*frac;
}

function drawTimeDomain(){
  var r=setupHiDPI(canvases.timedomain), ctx=r.ctx, w=r.w, h=r.h; if(w<=10) return;
  ctx.fillStyle='#faf9f7'; ctx.fillRect(0,0,w,h);
  var pad={l:40,r:12,t:10,b:28}, span=isIncoherent(state.shape)?120:Math.max(110,Math.min(340,Math.abs(state.tau)+state.tauP*2.4));
  var ax=drawAxes(ctx,w,h,pad,-span,span,-2.2,2.2,'时间 t (fs)','E(t)');
  var samples=600, tau=state.tau;
  ctx.strokeStyle='#1D9E75'; ctx.lineWidth=1.6; ctx.beginPath();
  for(var i=0;i<samples;i++){ var t=-span+(i/(samples-1))*(2*span), px=ax.mapX(t), py=ax.mapY(sampleFieldAt(t)+sampleFieldAt(t-tau)); if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py); }
  ctx.stroke();
}

var interferogramGeom=null;
function drawInterferogram(){
  var r=setupHiDPI(canvases.interferogram), ctx=r.ctx, w=r.w, h=r.h; if(w<=10) return;
  ctx.fillStyle='#faf9f7'; ctx.fillRect(0,0,w,h);
  var pad={l:40,r:12,t:10,b:28}, xMin=-state.tauMax, xMax=state.tauMax;
  var ax=drawAxes(ctx,w,h,pad,xMin,xMax,-0.1,2.15,'扫描延时 τ (fs)','I(τ)/I_DC');
  interferogramGeom={pad:pad,plotW:ax.plotW,xMin:xMin,xMax:xMax};
  var iStart=Math.max(0,Math.floor(xMin/DT)+HALF_N), iEnd=Math.min(N-1,Math.ceil(xMax/DT)+HALF_N);
  ctx.strokeStyle='#EF9F27'; ctx.lineWidth=1.5; ctx.beginPath();
  for(var i=iStart;i<=iEnd;i++){ var px=ax.mapX(tAxis[i]), py=ax.mapY(I_norm[i]); if(i===iStart) ctx.moveTo(px,py); else ctx.lineTo(px,py); }
  ctx.stroke();
  var currX=ax.mapX(state.tau), currIdx=Math.max(0,Math.min(N-1,Math.round(state.tau/DT)+HALF_N));
  ctx.strokeStyle='#7F77DD'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(currX,pad.t); ctx.lineTo(currX,pad.t+ax.plotH); ctx.stroke();
  ctx.fillStyle='#7F77DD'; ctx.beginPath(); ctx.arc(currX,ax.mapY(I_norm[currIdx]),4,0,Math.PI*2); ctx.fill();
}

function drawSpectrum(){
  var r=setupHiDPI(canvases.spectrum), ctx=r.ctx, w=r.w, h=r.h; if(w<=10) return;
  ctx.fillStyle='#faf9f7'; ctx.fillRect(0,0,w,h);
  var pad={l:40,r:12,t:10,b:28}, xMax=F_DISPLAY_MAX;
  var ax=drawAxes(ctx,w,h,pad,0,xMax,-0.25,1.15,'频率 f (THz)','归一化功率谱');
  var kMax=Math.min(HALF_N-1,Math.ceil(xMax/DF));
  ctx.strokeStyle='#7F77DD'; ctx.lineWidth=2; ctx.beginPath();
  for(var k=0;k<=kMax;k++){ var px=ax.mapX(k*DF), py=ax.mapY(S_ftir[k]); if(k===0) ctx.moveTo(px,py); else ctx.lineTo(px,py); }
  ctx.stroke();
  ctx.strokeStyle='#e11d48'; ctx.lineWidth=1.8; ctx.setLineDash([6,4]); ctx.beginPath();
  for(var k2=0;k2<=kMax;k2++){ var px2=ax.mapX(k2*DF), py2=ax.mapY(S_true[k2]); if(k2===0) ctx.moveTo(px2,py2); else ctx.lineTo(px2,py2); }
  ctx.stroke(); ctx.setLineDash([]);
}

function renderAll(){ if(!isSimActive) return; drawInterferometer(); drawTimeDomain(); drawInterferogram(); drawSpectrum(); }

function syncControlsUI(){
  var incoh=isIncoherent(state.shape), isBB=state.shape==='blackbody';
  els.ctrlTauP.style.display=isBB?'none':'flex'; els.ctrlF0.style.display=isBB?'none':'flex';
  els.ctrlTemp.style.display=isBB?'flex':'none'; els.ctrlIncoherent.style.display=incoh?'flex':'none';
  els.valTauP.textContent=state.tauP+' fs'; els.valF0.textContent=state.f0+' THz'; els.valTemp.textContent=state.tempK+' K';
  els.valTauMax.textContent=state.tauMax+' fs'; els.sliderTau.min=-state.tauMax; els.sliderTau.max=state.tauMax;
  state.tau=Math.max(-state.tauMax,Math.min(state.tauMax,state.tau));
  els.sliderTau.value=state.tau; els.valTau.textContent=(state.tau>=0?'+':'')+state.tau.toFixed(1)+' fs';
  document.querySelectorAll('.ftir-btn[data-shape]').forEach(function(b){ b.classList.remove('active-coh','active-inc'); if(b.dataset.shape===state.shape) b.classList.add(isIncoherent(b.dataset.shape)?'active-inc':'active-coh'); });
  document.querySelectorAll('.ftir-btn[data-win]').forEach(function(b){ b.classList.toggle('active-win',b.dataset.win===state.winType); });
  document.querySelectorAll('.ftir-btn[data-avg]').forEach(function(b){ b.classList.toggle('active-avg',Number(b.dataset.avg)===state.avgCount); });
  els.textScan.textContent=state.isScanning?'暂停扫描':'自动扫描';
  els.btnScan.classList.toggle('scanning',state.isScanning);
}

function refresh(){ syncControlsUI(); recomputePhysics(); renderAll(); }
document.querySelectorAll('[data-shape]').forEach(function(b){ b.addEventListener('click',function(){ state.shape=b.dataset.shape; refresh(); }); });
document.querySelectorAll('[data-win]').forEach(function(b){ b.addEventListener('click',function(){ state.winType=b.dataset.win; refresh(); }); });
document.querySelectorAll('[data-avg]').forEach(function(b){ b.addEventListener('click',function(){ state.avgCount=Number(b.dataset.avg); refresh(); }); });
els.sliderTauP.addEventListener('input',function(e){ state.tauP=Number(e.target.value); refresh(); });
els.sliderF0.addEventListener('input',function(e){ state.f0=Number(e.target.value); refresh(); });
els.sliderTemp.addEventListener('input',function(e){ state.tempK=Number(e.target.value); refresh(); });
els.sliderTauMax.addEventListener('input',function(e){ state.tauMax=Number(e.target.value); refresh(); });
els.sliderTau.addEventListener('input',function(e){ state.tau=Number(e.target.value); els.valTau.textContent=(state.tau>=0?'+':'')+state.tau.toFixed(1)+' fs'; updateMetricsUI(); drawInterferometer(); drawTimeDomain(); drawInterferogram(); });
els.btnReseed.addEventListener('click',function(){ state.rngSeed=(state.rngSeed+7919)>>>0; recomputePhysics(); renderAll(); });
els.btnScan.addEventListener('click',function(){ state.isScanning=!state.isScanning; syncControlsUI(); });
els.btnZeroTau.addEventListener('click',function(){ state.isScanning=false; state.tau=0; syncControlsUI(); updateMetricsUI(); renderAll(); });
els.btnReset.addEventListener('click',function(){ state={shape:'gaussian',tauP:30,f0:100,tempK:1500,avgCount:0,rngSeed:202605,tauMax:150,tau:0,winType:'boxcar',isScanning:false,scanDir:1,animTime:0}; refresh(); });
els.btnPresetRes.addEventListener('click',function(){ state.shape='dual'; state.tauP=45; state.f0=100; state.tauMax=30; state.tau=0; state.winType='boxcar'; state.isScanning=false; refresh(); });

var isDrag=false;
function scrubTau(cx){ if(!interferogramGeom) return; var rect=canvases.interferogram.getBoundingClientRect(), g=interferogramGeom; var ratio=Math.max(0,Math.min(1,(cx-rect.left-g.pad.l)/g.plotW)); state.tau=Math.round((g.xMin+ratio*(g.xMax-g.xMin))*2)/2; els.sliderTau.value=state.tau; els.valTau.textContent=(state.tau>=0?'+':'')+state.tau.toFixed(1)+' fs'; updateMetricsUI(); drawInterferometer(); drawTimeDomain(); drawInterferogram(); }
canvases.interferogram.addEventListener('mousedown',function(e){ isDrag=true; state.isScanning=false; syncControlsUI(); scrubTau(e.clientX); });
window.addEventListener('mousemove',function(e){ if(isDrag) scrubTau(e.clientX); });
window.addEventListener('mouseup',function(){ isDrag=false; });

var lastTs=performance.now();
function animateLoop(ts){
  var dtSec=Math.min(0.05,(ts-lastTs)/1000); lastTs=ts;
  if(isSimActive){
    state.animTime+=dtSec;
    if(state.isScanning){
      var speed=Math.max(20,(2*state.tauMax)/8);
      state.tau+=state.scanDir*speed*dtSec;
      if(state.tau>=state.tauMax){ state.tau=state.tauMax; state.scanDir=-1; }
      else if(state.tau<=-state.tauMax){ state.tau=-state.tauMax; state.scanDir=1; }
      els.sliderTau.value=state.tau; els.valTau.textContent=(state.tau>=0?'+':'')+state.tau.toFixed(1)+' fs';
      updateMetricsUI(); drawTimeDomain(); drawInterferogram();
    }
    drawInterferometer();
  }
  requestAnimationFrame(animateLoop);
}

window.addEventListener('resize', renderAll);
window.addEventListener('load', function(){ syncControlsUI(); recomputePhysics(); requestAnimationFrame(animateLoop); });
})();
