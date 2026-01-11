 const canvas = document.getElementById('network-canvas');
        const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
        const container = document.getElementById('canvasContainer');
        const layersInput = document.getElementById('layersInput');
        const neuronsContainer = document.getElementById('neuronsContainer');
        const layersValLabel = document.getElementById('layersVal');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const exportBtn = document.getElementById('exportBtn');
        const exportText = document.getElementById('exportText');
        const backpropToggle = document.getElementById('backpropToggle');
        const jitterToggle = document.getElementById('jitterToggle');
        const telemetryToggle = document.getElementById('telemetryToggle');
        const speedInput = document.getElementById('speedInput');
        const speedVal = document.getElementById('speedVal');
        const presetSelect = document.getElementById('presetSelect');
        const analysisStats = document.getElementById('analysisStats');
        const analysisOverlay = document.getElementById('analysisOverlay');
        const sparklineCanvas = document.getElementById('sparkline');
        const sctx = sparklineCanvas.getContext('2d');

        let width, height;
        let layers = [];
        let isPlaying = true;
        let globalTime = 0;
        let animationSpeed = 1.0;
        let layerConfig = [4, 6, 2]; 
        let layerNames = ["Input Vector", "Hidden Layer", "Output Logits"];
        let layerBaseValues = [0.8, 0.4, 0.2, 0.1, 0.1, 0.1];
        let layerActivations = ["none", "relu", "sigmoid", "relu", "relu", "relu"];
        let nodeBoosts = {}; 
        
        let entropyHistory = Array(50).fill(0.5);
        let currentTheme = { primary: '#3b82f6', accent: '#ef4444' };

        let mediaRecorder;
        let recordedChunks = [];

        function setTheme(type) {
            const root = document.documentElement;
            if (type === 'blue') {
                currentTheme = { primary: '#3b82f6', accent: '#ef4444' };
            } else if (type === 'cyberpunk') {
                currentTheme = { primary: '#d946ef', accent: '#22d3ee' };
            } else if (type === 'matrix') {
                currentTheme = { primary: '#10b981', accent: '#f59e0b' };
            }
            root.style.setProperty('--primary', currentTheme.primary);
            root.style.setProperty('--primary-glow', currentTheme.primary + '80');
        }

        const activate = (x, type) => {
            if (type === 'relu') return Math.max(0, x);
            if (type === 'sigmoid') return 1 / (1 + Math.exp(-x * 5));
            if (type === 'tanh') return Math.tanh(x * 2);
            return x;
        };

        class Neuron {
            constructor(x, y, layerIdx, neuronIdx) {
                this.x = x; this.y = y; this.layerIdx = layerIdx; this.neuronIdx = neuronIdx;
                this.radius = 16;
            }

            draw() {
                const totalCycle = (layerConfig.length + 1) * 2;
                const cycle = (globalTime * 0.8) % totalCycle;
                const isForward = cycle < (layerConfig.length + 1);
                const activeLayer = isForward ? Math.floor(cycle) : (layerConfig.length + 1) - (Math.floor(cycle) - (layerConfig.length + 1));
                
                const isActive = Math.floor(activeLayer) === this.layerIdx;
                const intensity = isActive ? 1 - (cycle % 1) : 0;

                let val = layerBaseValues[this.layerIdx] || 0.1;
                if (nodeBoosts[`${this.layerIdx}-${this.neuronIdx}`]) val = 1.0;

                const activatedVal = activate(val, layerActivations[this.layerIdx]);
                const visualRadius = this.radius + (activatedVal * 6);

                ctx.save();
                if (intensity > 0) {
                    ctx.shadowBlur = 30 * intensity;
                    ctx.shadowColor = isForward ? currentTheme.primary : currentTheme.accent;
                }

                ctx.beginPath();
                ctx.arc(this.x, this.y, visualRadius, 0, Math.PI * 2);
                ctx.fillStyle = intensity > 0.1 ? (isForward ? currentTheme.primary : currentTheme.accent) : '#1e293b';
                ctx.fill();
                
                ctx.strokeStyle = intensity > 0.1 ? '#ffffff' : '#334155';
                ctx.lineWidth = intensity > 0.1 ? 2.5 : 1;
                ctx.stroke();
                ctx.restore();
            }
        }

        class Connection {
            constructor(start, end) { this.start = start; this.end = end; }

            draw() {
                const totalCycle = (layerConfig.length + 1) * (backpropToggle.checked ? 2 : 1);
                const cycle = (globalTime * 0.8) % totalCycle;
                const isForward = cycle < (layerConfig.length + 1);
                
                const activeLayer = isForward ? Math.floor(cycle) : (layerConfig.length) - (Math.floor(cycle) - (layerConfig.length + 1));
                const isFlowing = activeLayer === (isForward ? this.start.layerIdx : this.end.layerIdx);
                const progress = cycle % 1;

                const cp1x = this.start.x + (this.end.x - this.start.x) * 0.5;
                const cp2x = this.start.x + (this.end.x - this.start.x) * 0.5;

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(this.start.x, this.start.y);
                ctx.bezierCurveTo(cp1x, this.start.y, cp2x, this.end.y, this.end.x, this.end.y);
                
                const jitter = (jitterToggle.checked ? Math.random() * 0.05 : 0);
                ctx.strokeStyle = isFlowing ? (isForward ? currentTheme.primary + '88' : currentTheme.accent + '88') : 'rgba(51, 65, 85, 0.15)';
                ctx.lineWidth = isFlowing ? 2.5 + jitter * 10 : 1;
                ctx.stroke();

                if (isFlowing && isPlaying) {
                    this.drawParticle(isForward ? progress : 1 - progress, cp1x, cp2x, isForward);
                }
                ctx.restore();
            }

            drawParticle(t, cp1x, cp2x, isForward) {
                const cx = Math.pow(1 - t, 3) * this.start.x + 3 * Math.pow(1 - t, 2) * t * cp1x + 3 * (1 - t) * Math.pow(t, 2) * cp2x + Math.pow(t, 3) * this.end.x;
                const cy = Math.pow(1 - t, 3) * this.start.y + 3 * Math.pow(1 - t, 2) * t * this.start.y + 3 * (1 - t) * Math.pow(t, 2) * this.end.y + Math.pow(t, 3) * this.end.y;
                ctx.beginPath();
                ctx.arc(cx, cy, 5, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffff';
                ctx.shadowBlur = 15;
                ctx.shadowColor = isForward ? currentTheme.primary : currentTheme.accent;
                ctx.fill();
            }
        }

        function updateAnalytics() {
            if (!telemetryToggle.checked) {
                analysisOverlay.classList.add('hidden-panel');
                return;
            } else {
                analysisOverlay.classList.remove('hidden-panel');
            }

            analysisStats.innerHTML = '';
            let totalActivity = 0;
            const jitter = jitterToggle.checked ? Math.random() * 0.1 : 0;
            document.getElementById('jitterVal').textContent = jitter.toFixed(3);

            layerNames.forEach((name, i) => {
                const base = layerBaseValues[i] || 0;
                const val = activate(base, layerActivations[i]) + jitter;
                totalActivity += val;
                
                const div = document.createElement('div');
                div.className = "space-y-1";
                div.innerHTML = `
                    <div class="flex justify-between text-[10px] mono">
                        <span class="text-slate-400 truncate w-24">${name}</span>
                        <span style="color: ${currentTheme.primary}">${val.toFixed(3)}</span>
                    </div>
                    <div class="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                        <div class="h-full transition-all duration-300" style="width: ${Math.min(100, val * 100)}%; background: ${currentTheme.primary}"></div>
                    </div>
                `;
                analysisStats.appendChild(div);
            });

            if (isPlaying) {
                entropyHistory.push((totalActivity / Math.max(1, layerConfig.length)));
                entropyHistory.shift();
                drawSparkline();
            }
        }

        function drawSparkline() {
            const rect = sparklineCanvas.getBoundingClientRect();
            if (sparklineCanvas.width !== rect.width || sparklineCanvas.height !== rect.height) {
                sparklineCanvas.width = rect.width;
                sparklineCanvas.height = rect.height;
            }

            const w = sparklineCanvas.width;
            const h = sparklineCanvas.height;
            
            sctx.clearRect(0, 0, w, h);
            sctx.beginPath();
            sctx.strokeStyle = currentTheme.primary;
            sctx.lineWidth = 2;
            sctx.lineCap = 'round';
            sctx.lineJoin = 'round';

            const step = w / (entropyHistory.length - 1);
            entropyHistory.forEach((val, i) => {
                const x = i * step;
                const y = h - (Math.min(1.2, Math.max(0, val)) * h * 0.7) - 4;
                if (i === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
            });
            sctx.stroke();

            sctx.lineTo(w, h); 
            sctx.lineTo(0, h);
            sctx.closePath();
            sctx.fillStyle = currentTheme.primary + '22';
            sctx.fill();
        }

        function buildNeuronControls() {
            neuronsContainer.innerHTML = '';
            layerConfig.forEach((count, i) => {
                const div = document.createElement('div');
                div.className = "group p-4 rounded-xl bg-slate-900/50 border border-white/5 hover:border-blue-500/30 transition-all";
                div.innerHTML = `
                    <div class="mb-3"><input type="text" value="${layerNames[i]}" class="w-full bg-transparent border-none p-0 text-[10px] font-bold text-slate-400 uppercase tracking-widest layer-name-input" data-index="${i}"></div>
                    <div class="grid grid-cols-2 gap-2 mb-3">
                        <select class="bg-slate-800 text-[9px] text-slate-300 rounded border-none py-1 px-1.5 layer-activation-input" data-index="${i}">
                            <option value="none" ${layerActivations[i] === 'none' ? 'selected' : ''}>Linear</option>
                            <option value="relu" ${layerActivations[i] === 'relu' ? 'selected' : ''}>ReLU</option>
                            <option value="sigmoid" ${layerActivations[i] === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                            <option value="tanh" ${layerActivations[i] === 'tanh' ? 'selected' : ''}>Tanh</option>
                        </select>
                        <input type="number" step="0.1" value="${layerBaseValues[i]}" class="bg-slate-800 text-slate-200 border-none rounded py-1 px-1.5 text-[10px] font-mono text-center layer-value-input" data-index="${i}">
                    </div>
                    <input type="range" min="1" max="12" value="${count}" class="w-full layer-neuron-input" data-index="${i}">
                `;
                neuronsContainer.appendChild(div);
            });

            document.querySelectorAll('.layer-neuron-input').forEach(input => {
                input.oninput = (e) => {
                    layerConfig[parseInt(e.target.dataset.index)] = parseInt(e.target.value);
                    initNetwork(); buildNeuronControls();
                };
            });
            document.querySelectorAll('.layer-name-input').forEach(input => {
                input.oninput = (e) => layerNames[parseInt(e.target.dataset.index)] = e.target.value;
            });
            document.querySelectorAll('.layer-value-input').forEach(input => {
                input.oninput = (e) => layerBaseValues[parseInt(e.target.dataset.index)] = parseFloat(e.target.value) || 0;
            });
            document.querySelectorAll('.layer-activation-input').forEach(input => {
                input.onchange = (e) => layerActivations[parseInt(e.target.dataset.index)] = e.target.value;
            });
        }

        function initNetwork() {
            layers = [];
            const xPadding = width * 0.15;
            const xStep = (width - xPadding * 2) / (layerConfig.length - 1 || 1);
            for (let i = 0; i < layerConfig.length; i++) {
                const count = layerConfig[i];
                const layerNeurons = [];
                const yPadding = height * 0.25; 
                const yStep = (height - yPadding * 2) / (count - 1 || 1);
                const startY = count === 1 ? height / 2 : yPadding;
                for (let j = 0; j < count; j++) {
                    layerNeurons.push(new Neuron(xPadding + xStep * i, startY + (count === 1 ? 0 : yStep * j), i, j));
                }
                layers.push(layerNeurons);
            }
        }

        function resize() {
            width = container.clientWidth; height = container.clientHeight;
            canvas.width = width * 2; 
            canvas.height = height * 2;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(2, 2);
            canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
            initNetwork();
        }

        function animate() {
            ctx.fillStyle = "#020617";
            ctx.fillRect(0, 0, width, height);
            
            if (isPlaying) globalTime += 0.025 * animationSpeed;

            ctx.save();
            ctx.textAlign = 'center'; ctx.font = '600 10px "JetBrains Mono"';
            layers.forEach((l, i) => {
                ctx.fillStyle = '#64748b'; 
                if (l[0]) ctx.fillText(layerNames[i].toUpperCase(), l[0].x, 40);
            });
            ctx.restore();

            for (let i = 0; i < layers.length - 1; i++) {
                for (let n1 of layers[i]) for (let n2 of layers[i + 1]) new Connection(n1, n2).draw();
            }
            layers.forEach(layer => layer.forEach(neuron => neuron.draw()));
            updateAnalytics();
            requestAnimationFrame(animate);
        }

        speedInput.oninput = (e) => {
            animationSpeed = parseFloat(e.target.value);
            speedVal.textContent = animationSpeed.toFixed(1) + 'x';
        };

        exportBtn.onclick = () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                return;
            }
            recordedChunks = [];
            const stream = canvas.captureStream(60); 
            const options = {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 10000000 
            };
            try { mediaRecorder = new MediaRecorder(stream, options); } catch (e) {
                mediaRecorder = new MediaRecorder(stream, { videoBitsPerSecond: 10000000 });
            }
            mediaRecorder.ondataavailable = (event) => { if (event.data.size > 0) recordedChunks.push(event.data); };
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none'; a.href = url;
                a.download = `neural-flow-HD-${Date.now()}.webm`;
                document.body.appendChild(a); a.click();
                setTimeout(() => { document.body.removeChild(a); window.URL.revokeObjectURL(url); }, 100);
                exportText.textContent = "Export HQ Video";
                exportBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
                document.getElementById('recordingIndicator').classList.add('hidden');
            };
            mediaRecorder.start();
            exportText.textContent = "Stop Recording";
            exportBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            document.getElementById('recordingIndicator').classList.remove('hidden');
        };

        presetSelect.onchange = (e) => {
            const mode = e.target.value;
            if (mode === 'bottleneck') {
                layerConfig = [4, 8, 2, 8, 4];
                layerNames = ["Input", "Encoder", "Bottleneck", "Decoder", "Output"];
                layerActivations = ["none", "relu", "tanh", "relu", "sigmoid"];
            } else if (mode === 'wide') {
                layerConfig = [2, 12, 2];
                layerNames = ["Input", "Wide Feature Map", "Output"];
                layerActivations = ["none", "relu", "sigmoid"];
            } else if (mode === 'classifier') {
                layerConfig = [5, 4, 4, 1];
                layerNames = ["Data", "L1", "L2", "Probability"];
                layerActivations = ["none", "relu", "relu", "sigmoid"];
            }
            layersInput.value = layerConfig.length;
            layersValLabel.textContent = layerConfig.length + ' Layers';
            buildNeuronControls(); initNetwork();
        };

        layersInput.oninput = (e) => {
            const val = parseInt(e.target.value);
            layersValLabel.textContent = val + ' Layers';
            while (layerConfig.length < val) { 
                layerConfig.push(4); layerNames.push(`L${layerConfig.length}`); 
                layerBaseValues.push(0.3); layerActivations.push('relu');
            }
            layerConfig = layerConfig.slice(0, val);
            buildNeuronControls(); initNetwork();
        };

        playPauseBtn.onclick = () => {
            isPlaying = !isPlaying;
            document.getElementById('playIcon').textContent = isPlaying ? '⏸' : '▶';
            document.getElementById('playText').textContent = isPlaying ? 'Pause' : 'Play';
            document.getElementById('statusIndicator').className = isPlaying ? 'w-2 h-2 bg-emerald-500 rounded-full animate-pulse' : 'w-2 h-2 bg-slate-600 rounded-full';
        };

        canvas.onclick = (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            layers.forEach(layer => layer.forEach(n => {
                const dist = Math.hypot(n.x - mouseX, n.y - mouseY);
                if (dist < 20) {
                    const id = `${n.layerIdx}-${n.neuronIdx}`;
                    nodeBoosts[id] = true;
                    setTimeout(() => delete nodeBoosts[id], 2000);
                }
            }));
        };

        window.addEventListener('resize', () => { resize(); drawSparkline(); });
        window.onload = () => { resize(); buildNeuronControls(); animate(); setTheme('blue'); };
