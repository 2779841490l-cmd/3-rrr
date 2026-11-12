import json
import os

with open("experiments/aim_visualization_data.json", "r", encoding="utf-8") as f:
    raw_json = f.read()

artifact_dir = r"C:\Users\27798\.gemini\antigravity\brain\44065d9b-d09c-40b8-901e-c33219ae492a"
os.makedirs(artifact_dir, exist_ok=True)
out_path = os.path.join(artifact_dir, "aim_benchmark_visualizer.html")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3-RRR 100ms 3200DPI FPS 打靶精度检验平台</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ background: transparent; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    .tab-btn.active {{ background-color: #0284c7; color: #ffffff; border-color: #38bdf8; }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2 m-0">
  <div class="space-y-3">
    <!-- Top Bar -->
    <div class="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-900/90 rounded-xl border border-slate-800 shadow-lg">
      <div class="flex items-center gap-3">
        <span class="text-lg">🎯</span>
        <div>
          <h1 class="text-sm font-bold text-sky-400">3-RRR 打靶精度与 R 曲线离线检验平台</h1>
          <p class="text-xs text-slate-400">100ms 百毫秒极速弹道 · 120 发实机数据集检验</p>
        </div>
      </div>

      <div class="flex items-center gap-2 text-xs">
        <label class="text-slate-400 font-semibold">场次选择:</label>
        <select id="runSelect" class="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs outline-none focus:border-sky-500">
          <option value="POOLED_CORE">★ 核心基准 (6 场 120 发合并, 均误差 32.8 um)</option>
          <option value="_default_r1">★ 最佳单场 _default_r1 (20 发 100% @ 14px)</option>
          <option value="_tel_r1">遥测基准 _tel_r1 (max_walk=4, trials=6.4)</option>
          <option value="_w6_r1">消融对照 _w6_r1 (max_walk=6, trials=8.4)</option>
          <option value="_base">场次 _base (19/20)</option>
          <option value="_mw_r1">加权实验 _mw_r1 (19/20)</option>
          <option value="_mw_r2">加权实验 _mw_r2 (18/20)</option>
          <option value="_nomw_r2">无加权对照 _nomw_r2 (20/20)</option>
          <option value="_nomw_r3">无加权对照 _nomw_r3 (19/20)</option>
        </select>
      </div>

      <div class="flex items-center gap-2 text-xs">
        <label class="text-slate-400 font-semibold">目标球半径 R:</label>
        <input id="rSlider" type="range" min="4" max="32" step="0.5" value="18" class="w-28 accent-sky-400">
        <span id="rValReadout" class="font-mono font-bold text-amber-300 w-32 text-right">R = 18.0 px</span>
      </div>

      <div class="flex items-center gap-1 text-xs">
        <button onclick="setRadius(9)" class="px-2 py-1 bg-red-950/60 text-red-300 border border-red-800/80 rounded hover:bg-red-900/60">物理死区 9px</button>
        <button onclick="setRadius(14)" class="px-2 py-1 bg-emerald-950/60 text-emerald-300 border border-emerald-800/80 rounded hover:bg-emerald-900/60">最佳 14px (28px直)</button>
        <button onclick="setRadius(18)" class="px-2 py-1 bg-sky-950/60 text-sky-300 border border-sky-800/80 rounded hover:bg-sky-900/60">规格 18px (36px直)</button>
        <button onclick="setRadius(24)" class="px-2 py-1 bg-slate-800 text-slate-300 border border-slate-700 rounded hover:bg-slate-700">宽松 24px</button>
      </div>
    </div>

    <!-- Metrics Cards Banner -->
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs font-mono">
      <div class="p-2.5 bg-slate-900/80 rounded-lg border border-slate-800 flex flex-col justify-between">
        <span class="text-slate-400 text-[11px]">命中率 (当前 R)</span>
        <div class="flex items-baseline justify-between mt-1">
          <span id="hitRateText" class="text-base font-bold text-emerald-400">95.8%</span>
          <span id="hitRatioText" class="text-slate-400 text-[11px]">115 / 120</span>
        </div>
      </div>
      <div class="p-2.5 bg-slate-900/80 rounded-lg border border-slate-800 flex flex-col justify-between">
        <span class="text-slate-400 text-[11px]">平均光学误差</span>
        <div class="flex items-baseline justify-between mt-1">
          <span id="meanOptErrText" class="text-base font-bold text-sky-400">4.13 cnt</span>
          <span id="meanUmText" class="text-slate-400 text-[11px]">32.8 um</span>
        </div>
      </div>
      <div class="p-2.5 bg-slate-900/80 rounded-lg border border-slate-800 flex flex-col justify-between">
        <span class="text-slate-400 text-[11px]">中位数误差 (50%半径)</span>
        <div class="flex items-baseline justify-between mt-1">
          <span id="medianErrText" class="text-base font-bold text-purple-400">3.16 cnt</span>
          <span id="medianPxText" class="text-slate-400 text-[11px]">6.0 px</span>
        </div>
      </div>
      <div class="p-2.5 bg-slate-900/80 rounded-lg border border-slate-800 flex flex-col justify-between">
        <span class="text-slate-400 text-[11px]">p90 误差尾部</span>
        <div class="flex items-baseline justify-between mt-1">
          <span id="p90ErrText" class="text-base font-bold text-amber-400">8.57 cnt</span>
          <span id="maxErrText" class="text-slate-400 text-[11px]">max 14.5 cnt</span>
        </div>
      </div>
      <div class="p-2.5 bg-slate-900/80 rounded-lg border border-slate-800 flex flex-col justify-between">
        <span class="text-slate-400 text-[11px]">平均弹道时间</span>
        <div class="flex items-baseline justify-between mt-1">
          <span id="meanTimeText" class="text-base font-bold text-cyan-400">92.0 ms</span>
          <span class="text-slate-400 text-[11px]">&lt; 100ms 达标</span>
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex gap-1.5 border-b border-slate-800 pb-1 text-xs">
      <button class="tab-btn active px-3 py-1.5 rounded-t-lg font-semibold border border-transparent transition-colors" onclick="switchTab('arenaTab', this)">🎯 2D 靶场落点全景</button>
      <button class="tab-btn px-3 py-1.5 rounded-t-lg font-semibold text-slate-400 border border-transparent transition-colors hover:text-slate-200" onclick="switchTab('rCurveTab', this)">📈 R 曲线 (Hit Rate CDF)</button>
      <button class="tab-btn px-3 py-1.5 rounded-t-lg font-semibold text-slate-400 border border-transparent transition-colors hover:text-slate-200" onclick="switchTab('scatterTab', this)">📊 误差散布 (dx vs dy)</button>
      <button class="tab-btn px-3 py-1.5 rounded-t-lg font-semibold text-slate-400 border border-transparent transition-colors hover:text-slate-200" onclick="switchTab('conclusionTab', this)">🧪 否证分析与硬件定案</button>
    </div>

    <!-- Tab 1: 2D Target Arena -->
    <div id="arenaTab" class="tab-content block">
      <div class="relative bg-slate-950 rounded-xl border border-slate-800 overflow-hidden shadow-inner p-2">
        <div class="flex items-center justify-between px-2 py-1 text-xs text-slate-400 border-b border-slate-900">
          <span>全屏 1536x864 坐标映射 · 3-RRR 安全包络 [±280 cnt, ±210 cnt / ±560 px, ±300 px]</span>
          <div class="flex items-center gap-3">
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span> 命中 (误差 ≤ R)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-red-500 inline-block"></span> 脱靶 (误差 &gt; R)</span>
            <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-slate-400 inline-block"></span> 误差连线</span>
          </div>
        </div>
        <canvas id="arenaCanvas" width="920" height="420" class="w-full h-auto cursor-crosshair block"></canvas>
        <div id="hudTooltip" class="absolute hidden pointer-events-none bg-slate-900/95 border border-sky-500/60 rounded-lg p-2.5 text-xs font-mono shadow-2xl backdrop-blur-md z-10"></div>
      </div>
    </div>

    <!-- Tab 2: R Curve -->
    <div id="rCurveTab" class="tab-content hidden">
      <div class="bg-slate-950 rounded-xl border border-slate-800 p-4 shadow-inner">
        <div class="flex items-center justify-between mb-2 text-xs text-slate-300">
          <span class="font-bold text-sky-400">目标球半径 R 经验累积命中分布 (Empirical Hit Rate CDF)</span>
          <span class="text-slate-400">横轴: 半径 R (px) | 纵轴: 命中率 (%)</span>
        </div>
        <canvas id="rCurveCanvas" width="920" height="380" class="w-full h-auto block"></canvas>
      </div>
    </div>

    <!-- Tab 3: Residual Scatter -->
    <div id="scatterTab" class="tab-content hidden">
      <div class="bg-slate-950 rounded-xl border border-slate-800 p-4 shadow-inner">
        <div class="flex items-center justify-between mb-2 text-xs text-slate-300">
          <span class="font-bold text-sky-400">落点残差散布 (dx vs dy counts) &amp; 95% 置信椭圆</span>
          <span class="text-slate-400">dx: -0.54 cnt (t=-1.62 不显著) | dy: -0.06 cnt (t=-0.18 等于0)</span>
        </div>
        <canvas id="scatterCanvas" width="920" height="380" class="w-full h-auto block"></canvas>
      </div>
    </div>

    <!-- Tab 4: Conclusions -->
    <div id="conclusionTab" class="tab-content hidden">
      <div class="bg-slate-900/90 rounded-xl border border-slate-800 p-4 space-y-3 text-xs leading-relaxed">
        <div class="p-3 bg-sky-950/40 border border-sky-800/60 rounded-lg">
          <h3 class="font-bold text-sky-300 mb-1">📌 核心结论与实测定数</h3>
          <ul class="list-disc list-inside space-y-1 text-slate-300">
            <li><strong class="text-emerald-400">最佳单场规格:</strong> 100% 命中 R=14 px (直径 28 px / 111.1 um), 均光学误差 3.60 cnt (28.6 um)。</li>
            <li><strong class="text-sky-400">稳态保守规格:</strong> 95.8% 命中 R=18 px (直径 36 px / 142.9 um), 均光学误差 32.8 um, 均弹道 92.0 ms (&lt; 100ms 铁律成立)。</li>
            <li><strong class="text-red-400">物理死区地板:</strong> R &lt; 9 px 任何控制器都无法保证命中 (MG996R 内部死区 6~14 us / 晶格量化产生 4.5 cnt 物理台阶)。</li>
            <li><strong class="text-purple-400">中位数精度:</strong> 50% 命中半径仅 6 px (47.6 um), 说明多数目标落点极准，均值主要被尾部 10% 拉高。</li>
          </ul>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <h4 class="font-bold text-amber-400 mb-1">❌ 否证 1: 目标函数屏幕加权对齐 (metric_w)</h4>
            <p class="text-slate-300">直觉上按屏幕 X:Y = 2.0:1.4286 加权对齐评价指标应能白赚精度。但实测 <strong>ON 37/40 (92.5%) vs OFF 59/60 (98.3%)</strong>，均屏幕误差 7.72 vs 6.50 px。在非线性离散晶格上，加权会扭曲局部梯度择优，加权更近不等于落点更好。已坚决弃用，回归 OFF。</p>
          </div>
          <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <h4 class="font-bold text-amber-400 mb-1">❌ 否证 2: 放宽双侧晶格精修预算 (--dual-max-walk)</h4>
            <p class="text-slate-300">遥测显示 walk=4 时 12/20 发走完上限未收敛。单变量放宽至 walk=6 测试后：精修增益虽微涨 +1.33 cnt，但多次小步冲击产生持续静摩擦漂移，整发误差从 3.60 cnt 恶化至 4.85 cnt。已锁定 walk=4，不再盲目加步。</p>
          </div>
        </div>

        <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
          <h4 class="font-bold text-cyan-400 mb-1">🔧 为什么纯软件到顶了？下一步怎么办？</h4>
          <p class="text-slate-300">交叉统计证明 dx 均偏 -0.54 cnt (不显著), dy 均偏 -0.06 cnt (等于0)，且接近方向对称性差仅 0.28 cnt。残余 4.13 cnt 的本质是 MG996R 模拟比较器死区与齿轮间隙的随机粘滑跳变，无任何全局偏置可消。<strong>换用 STS3215 12 位磁编码器总线舵机（分辨率 0.088°）是跨入 1~2 counts 极致精度的唯一物理路径。</strong></p>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATASET = {raw_json};
    let currentRunKey = 'POOLED_CORE';
    let currentR = 18.0;
    let hoveredShot = null;

    function getActiveRows() {{
      if (currentRunKey === 'POOLED_CORE') return DATASET.pooled;
      return DATASET.runs[currentRunKey] || DATASET.pooled;
    }}

    function setRadius(r) {{
      document.getElementById('rSlider').value = r;
      updateRadius(r);
    }}

    function updateRadius(r) {{
      currentR = parseFloat(r);
      const um = (currentR * (7.9375 / 2.0)).toFixed(1);
      document.getElementById('rValReadout').innerText = `R = ${{currentR.toFixed(1)}} px (${{(2*currentR).toFixed(1)}}px直 / ${{um}}um)`;
      updateMetrics();
      renderArena();
      renderRCurve();
    }}

    function updateMetrics() {{
      const rows = getActiveRows();
      if (!rows.length) return;
      const hits = rows.filter(r => r.err_px <= currentR).length;
      const hitRate = ((100.0 * hits) / rows.length).toFixed(1);

      document.getElementById('hitRateText').innerText = `${{hitRate}}%`;
      document.getElementById('hitRatioText').innerText = `${{hits}} / ${{rows.length}}`;

      const errCnts = rows.map(r => r.err_cnt).sort((a,b)=>a-b);
      const errUms = rows.map(r => r.err_um);
      const times = rows.map(r => r.ms);

      const meanOpt = (errCnts.reduce((a,b)=>a+b,0)/rows.length).toFixed(2);
      const meanUm = (errUms.reduce((a,b)=>a+b,0)/rows.length).toFixed(1);
      const medianCnt = errCnts[Math.floor(errCnts.length*0.5)].toFixed(2);
      const medianPx = (rows.map(r=>r.err_px).sort((a,b)=>a-b)[Math.floor(rows.length*0.5)]).toFixed(1);
      const p90Cnt = errCnts[Math.floor(errCnts.length*0.9)].toFixed(2);
      const maxCnt = errCnts[errCnts.length-1].toFixed(1);
      const meanTime = (times.reduce((a,b)=>a+b,0)/rows.length).toFixed(1);

      document.getElementById('meanOptErrText').innerText = `${{meanOpt}} cnt`;
      document.getElementById('meanUmText').innerText = `${{meanUm}} um`;
      document.getElementById('medianErrText').innerText = `${{medianCnt}} cnt`;
      document.getElementById('medianPxText').innerText = `${{medianPx}} px`;
      document.getElementById('p90ErrText').innerText = `${{p90Cnt}} cnt`;
      document.getElementById('maxErrText').innerText = `max ${{maxCnt}} cnt`;
      document.getElementById('meanTimeText').innerText = `${{meanTime}} ms`;
    }}

    function switchTab(tabId, btn) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab-btn').forEach(b => {{
        b.classList.remove('active', 'text-white');
        b.classList.add('text-slate-400');
      }});
      document.getElementById(tabId).classList.remove('hidden');
      btn.classList.add('active', 'text-white');
      btn.classList.remove('text-slate-400');

      if (tabId === 'arenaTab') renderArena();
      if (tabId === 'rCurveTab') renderRCurve();
      if (tabId === 'scatterTab') renderScatter();
    }}

    // 2D Arena Canvas
    function renderArena() {{
      const cvs = document.getElementById('arenaCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      const cx = w / 2;
      const cy = h / 2;

      ctx.clearRect(0, 0, w, h);

      // Coordinate scaling: screen [-560, 560] x [-300, 300] to canvas
      const scaleX = (w * 0.44) / 560.0;
      const scaleY = (h * 0.44) / 300.0;

      // Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(0, cy); ctx.lineTo(w, cy);
      ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
      ctx.stroke();

      // Safe Envelope box
      const bx0 = cx - 560 * scaleX;
      const by0 = cy - 300 * scaleY;
      const bw = 1120 * scaleX;
      const bh = 600 * scaleY;
      ctx.strokeStyle = '#334155';
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(bx0, by0, bw, bh);
      ctx.setLineDash([]);

      const rows = getActiveRows();
      const rPx = currentR;
      const rCanX = rPx * scaleX;
      const rCanY = rPx * scaleY;

      rows.forEach(r => {{
        const tx = cx + r.stx * scaleX;
        const ty = cy + r.sty * scaleY;
        const ax = cx + r.sax * scaleX;
        const ay = cy + r.say * scaleY;

        const isHit = r.err_px <= rPx;

        // Ball tolerance ellipse
        ctx.fillStyle = isHit ? 'rgba(16, 185, 129, 0.18)' : 'rgba(239, 68, 68, 0.16)';
        ctx.strokeStyle = isHit ? '#10b981' : '#ef4444';
        ctx.lineWidth = (hoveredShot && hoveredShot.bid === r.bid) ? 2 : 1;

        ctx.beginPath();
        ctx.ellipse(tx, ty, rCanX, rCanY, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Target center
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(tx - 3, ty); ctx.lineTo(tx + 3, ty);
        ctx.moveTo(tx, ty - 3); ctx.lineTo(tx, ty + 3);
        ctx.stroke();

        // Error line
        ctx.strokeStyle = isHit ? 'rgba(52, 211, 153, 0.5)' : 'rgba(248, 113, 113, 0.6)';
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(tx, ty); ctx.lineTo(ax, ay);
        ctx.stroke();
        ctx.setLineDash([]);

        // Landing marker
        ctx.fillStyle = isHit ? '#22c55e' : '#facc15';
        ctx.beginPath();
        ctx.arc(ax, ay, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }});
    }}

    // R-Curve Canvas
    function renderRCurve() {{
      const cvs = document.getElementById('rCurveCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      const padL = 60, padR = 30, padT = 30, padB = 40;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      // Axes & grid
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      for (let y = 0; y <= 100; y += 20) {{
        const py = padT + plotH - (y / 100.0) * plotH;
        ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + plotW, py); ctx.stroke();
        ctx.fillStyle = '#64748b';
        ctx.font = '10px monospace';
        ctx.fillText(`${{y}}%`, padL - 35, py + 3);
      }}
      for (let x = 0; x <= 32; x += 4) {{
        const px = padL + (x / 32.0) * plotW;
        ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT + plotH); ctx.stroke();
        ctx.fillStyle = '#64748b';
        ctx.font = '10px monospace';
        ctx.fillText(`${{x}}px`, px - 10, padT + plotH + 16);
      }}
      ctx.setLineDash([]);

      const rows = getActiveRows();
      const errs = rows.map(r => r.err_px);

      // Draw pooled CDF
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let r = 2; r <= 32; r += 0.2) {{
        const hitPct = (100.0 * errs.filter(e => e <= r).length) / rows.length;
        const px = padL + (r / 32.0) * plotW;
        const py = padT + plotH - (hitPct / 100.0) * plotH;
        if (r === 2) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }}
      ctx.stroke();

      // Deadband limit line 9px
      const p9x = padL + (9.0 / 32.0) * plotW;
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(p9x, padT); ctx.lineTo(p9x, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#ef4444'; ctx.font = '10px monospace';
      ctx.fillText('死区极限 9px', p9x + 4, padT + 18);

      // Best 14px line
      const p14x = padL + (14.0 / 32.0) * plotW;
      ctx.strokeStyle = '#10b981'; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(p14x, padT); ctx.lineTo(p14x, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#10b981';
      ctx.fillText('最佳 100% (14px)', p14x + 4, padT + 34);

      // Conservative 18px line
      const p18x = padL + (18.0 / 32.0) * plotW;
      ctx.strokeStyle = '#facc15'; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(p18x, padT); ctx.lineTo(p18x, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#facc15';
      ctx.fillText('规格 95.8% (18px)', p18x + 4, padT + 50);

      // Current R indicator
      const curX = padL + (currentR / 32.0) * plotW;
      const curHit = (100.0 * errs.filter(e => e <= currentR).length) / rows.length;
      const curY = padT + plotH - (curHit / 100.0) * plotH;

      ctx.strokeStyle = '#ec4899'; ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(curX, padT); ctx.lineTo(curX, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#ec4899';
      ctx.beginPath(); ctx.arc(curX, curY, 5, 0, Math.PI * 2); ctx.fill();
    }}

    // Residual Scatter Canvas
    function renderScatter() {{
      const cvs = document.getElementById('scatterCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const scale = (w * 0.38) / 16.0;

      // Axes
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy); ctx.lineTo(w, cy);
      ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
      ctx.stroke();

      const rows = getActiveRows();
      rows.forEach(r => {{
        const dx = r.ax - r.tx;
        const dy = r.ay - r.ty;
        const px = cx + dx * scale;
        const py = cy + dy * scale;

        ctx.fillStyle = 'rgba(56, 189, 248, 0.7)';
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }});

      // 95% confidence circle/ellipse approximation
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx - 0.54 * scale, cy - 0.06 * scale, 8.8 * scale, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#ef4444';
      ctx.font = '11px monospace';
      ctx.fillText('+ 均值偏置: (-0.54, -0.06) cnt (零均值对称)', cx + 15, cy - 15);
    }}

    // Arena Mouse Interaction
    const arenaCvs = document.getElementById('arenaCanvas');
    const hud = document.getElementById('hudTooltip');

    arenaCvs.addEventListener('mousemove', e => {{
      const rect = arenaCvs.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (arenaCvs.width / rect.width);
      const my = (e.clientY - rect.top) * (arenaCvs.height / rect.height);

      const cx = arenaCvs.width / 2;
      const cy = arenaCvs.height / 2;
      const scaleX = (arenaCvs.width * 0.44) / 560.0;
      const scaleY = (arenaCvs.height * 0.44) / 300.0;

      const rows = getActiveRows();
      let found = null;
      let minD = 22;

      rows.forEach(r => {{
        const tx = cx + r.stx * scaleX;
        const ty = cy + r.sty * scaleY;
        const d = Math.hypot(mx - tx, my - ty);
        if (d < minD) {{
          minD = d;
          found = r;
        }}
      }});

      hoveredShot = found;
      if (found) {{
        hud.classList.remove('hidden');
        hud.style.left = `${{e.clientX - rect.left + 15}}px`;
        hud.style.top = `${{e.clientY - rect.top + 15}}px`;
        const isHit = found.err_px <= currentR;
        hud.innerHTML = `
          <div class="font-bold ${{isHit ? 'text-emerald-400' : 'text-red-400'}} mb-1">
            ${{isHit ? '✅ 命中 TARGET HIT' : '❌ 脱靶 TARGET MISS'}} (球 #${{found.bid}})
          </div>
          <div class="text-slate-300 space-y-0.5">
            <div>目标坐标: (${{found.stx.toFixed(0)}}, ${{found.sty.toFixed(0)}}) px</div>
            <div>实机落点: (${{found.sax.toFixed(0)}}, ${{found.say.toFixed(0)}}) px</div>
            <div>光学误差: <span class="text-amber-300 font-bold">${{found.err_cnt.toFixed(2)}} cnt</span> (${{found.err_um.toFixed(1)}} um / ${{found.err_px.toFixed(1)}} px)</div>
            <div>极速弹道: ${{found.ms.toFixed(1)}} ms</div>
            <div>精修步数: ${{found.trials}} 步</div>
          </div>
        `;
      }} else {{
        hud.classList.add('hidden');
      }}
      renderArena();
    }});

    arenaCvs.addEventListener('mouseleave', () => {{
      hud.classList.add('hidden');
      hoveredShot = null;
      renderArena();
    }});

    document.getElementById('rSlider').addEventListener('input', e => updateRadius(e.target.value));
    document.getElementById('runSelect').addEventListener('change', e => {{
      currentRunKey = e.target.value;
      updateMetrics();
      renderArena();
      renderRCurve();
      renderScatter();
    }});

    // Init
    updateRadius(18);
    renderArena();
  </script>
</body>
</html>"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Generated {out_path} ({len(html)} bytes)")

