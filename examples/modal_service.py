"""
Chat with the tutor on Modal using Qwen3-8B with true streaming memory.

Uses Modal's class pattern with keep_warm=1 to always have a container ready.
Model loads once at container start, not per-request.

Run:
    modal deploy examples/modal_chat.py

Then open the URL in your browser.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("streaming-memory")

MODEL_ID = "Qwen/Qwen3-8B"

memories_path = Path(__file__).parent / "memories.json"
dad_memories_path = Path(__file__).parent / "dad_memories.json"
package_path = Path(__file__).parent.parent / "streaming_memory"


def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "openai",
        "fastapi",
        "uvicorn",
        "pydantic",
    )
    .run_function(download_model)
    .add_local_file(memories_path, "/app/memories.json")
    .add_local_file(dad_memories_path, "/app/dad_memories.json")
    .add_local_dir(package_path, "/app/streaming_memory")
)


SCENARIOS = {
    "tutor": {
        "system_prompt": """You are an AI tutor who has been working with Alex, a Grade 5 student, for several months.

You have built up memories and insights about how he learns, what works for him, and your relationship.

When memories are provided, use them naturally to inform your responses. You know Alex well.

Think step by step in <think>...</think> tags before responding.

Speak as yourself - the tutor who has this history with Alex.""",
        "memory_file": "/app/memories.json",
        "memory_prefix": "[My memories from working with Alex:]",
    },
    "dad": {
        "system_prompt": """You are a helpful personal assistant who has access to the user's memories and notes.

You help them think through decisions by drawing on what you know about their life, relationships, and past experiences.

When memories are provided, use them naturally to inform your responses. Make connections between different memories when relevant.

Think step by step in <think>...</think> tags before responding.

Be warm and helpful, like a thoughtful friend who knows them well.""",
        "memory_file": "/app/dad_memories.json",
        "memory_prefix": "[User's memories and notes:]",
    },
}


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Streaming Memory</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'SF Mono', 'Fira Code', monospace; 
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 20px; 
            background: #0d1117; 
            color: #c9d1d9; 
            min-height: 100vh;
        }
        h1 { color: #58a6ff; margin-bottom: 8px; font-size: 1.5em; }
        .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 0.85em; }
        
        #chat { 
            height: calc(100vh - 200px); 
            overflow-y: auto; 
            border: 1px solid #30363d; 
            border-radius: 6px; 
            padding: 15px; 
            margin-bottom: 15px; 
            background: #161b22; 
        }
        
        .message-container { margin: 20px 0; }
        
        .timing-panel {
            font-size: 0.7em;
            color: #8b949e;
            margin-bottom: 8px;
            padding: 6px 10px;
            background: #21262d;
            border-radius: 4px;
            font-family: monospace;
        }
        .timing-panel span { margin-right: 12px; }
        .timing-panel .slow { color: #f85149; }
        .timing-panel .medium { color: #d29922; }
        .timing-panel .fast { color: #238636; }
        
        .memories-panel {
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 0.75em;
            max-height: 250px;
            overflow-y: auto;
        }
        .memories-header { color: #f0883e; font-weight: 600; margin-bottom: 6px; }
        .memory-item {
            color: #8b949e;
            padding: 6px 8px;
            border-left: 2px solid #30363d;
            margin: 4px 0;
            transition: all 0.3s ease;
        }
        .memory-item.new { border-left-color: #238636; color: #c9d1d9; background: rgba(35, 134, 54, 0.15); }
        .memory-item.removed { border-left-color: #f85149; opacity: 0.5; text-decoration: line-through; }
        
        .thinking-panel {
            background: #1a1f25;
            border: 1px solid #8b5cf6;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 0.8em;
            max-height: 400px;
            overflow-y: auto;
        }
        .thinking-header { color: #8b5cf6; font-weight: 600; margin-bottom: 6px; }
        .thinking-content { color: #7d8590; font-style: italic; white-space: pre-wrap; }
        
        .message { padding: 12px 16px; border-radius: 6px; line-height: 1.6; }
        .user { background: #21262d; border: 1px solid #30363d; }
        .assistant { background: transparent; color: #c9d1d9; }
        .assistant .cursor {
            display: inline-block; width: 8px; height: 16px;
            background: #58a6ff; animation: blink 1s infinite;
            vertical-align: text-bottom; margin-left: 2px;
        }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        
        #input-area { display: flex; gap: 10px; }
        #message { 
            flex: 1; padding: 12px; border: 1px solid #30363d; border-radius: 6px; 
            background: #21262d; color: #c9d1d9; font-size: 14px; font-family: inherit;
        }
        #message:focus { outline: none; border-color: #58a6ff; }
        button { 
            padding: 12px 24px; background: #238636; color: #fff; 
            border: none; border-radius: 6px; cursor: pointer; font-weight: 600;
        }
        button:hover { background: #2ea043; }
        button:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        
        .scrubber-panel { background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-top: 12px; }
        .scrubber-header { color: #58a6ff; font-weight: 600; margin-bottom: 8px; font-size: 0.85em; }
        .scrubber-slider { width: 100%; margin: 8px 0; -webkit-appearance: none; background: #30363d; height: 6px; border-radius: 3px; }
        .scrubber-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; background: #58a6ff; border-radius: 50%; cursor: pointer; }
        .scrubber-info { display: flex; justify-content: space-between; font-size: 0.75em; color: #8b949e; margin-bottom: 8px; }
        .scrubber-token { font-family: monospace; background: #161b22; padding: 4px 8px; border-radius: 4px; margin-bottom: 8px; }
        .scrubber-token .current { background: #238636; color: #fff; padding: 2px 4px; border-radius: 2px; }
        .scrubber-memories { font-size: 0.75em; max-height: 150px; overflow-y: auto; }
        .playback-controls { display: flex; gap: 8px; margin-bottom: 8px; }
        .playback-controls button { padding: 4px 12px; font-size: 0.8em; background: #30363d; }
    </style>
</head>
<body>
    <h1>🧠 Streaming Memory</h1>
    <p class="subtitle">Qwen3-8B on Modal • Memory re-retrieval every token • Context updates during generation</p>
    
    <div id="chat"></div>
    
    <div id="input-area">
        <input type="text" id="message" placeholder="Ask about Alex..." autofocus>
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
    
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('message');
        const sendBtn = document.getElementById('send-btn');
        let history = [];
        
        input.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !sendBtn.disabled) sendMessage(); });
        
        async function sendMessage() {
            const msg = input.value.trim();
            if (!msg) return;
            
            chat.innerHTML += `<div class="message-container"><div class="message user">${escapeHtml(msg)}</div></div>`;
            input.value = '';
            sendBtn.disabled = true;
            chat.scrollTop = chat.scrollHeight;
            
            const responseContainer = document.createElement('div');
            responseContainer.className = 'message-container';
            responseContainer.innerHTML = `
                <div class="timing-panel">⏱️ Initializing...</div>
                <div class="memories-panel"><div class="memories-header">📚 Retrieving memories...</div></div>
                <div class="thinking-panel" style="display:none"><div class="thinking-header">💭 Thinking...</div><div class="thinking-content"></div></div>
                <div class="message assistant"><span class="cursor"></span></div>
            `;
            chat.appendChild(responseContainer);
            
            const timingPanel = responseContainer.querySelector('.timing-panel');
            const memoriesPanel = responseContainer.querySelector('.memories-panel');
            const thinkingPanel = responseContainer.querySelector('.thinking-panel');
            const assistantMsg = responseContainer.querySelector('.assistant');
            
            const startTime = Date.now();
            let timing = {}, memoryChanges = 0, fullResponse = '', fullThinking = '', firstToken = false;
            
            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: msg, history })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, {stream: true});
                    // SSE format: "data: {...}\\n\\n"
                    const events = buffer.split('\\n\\n');
                    buffer = events.pop(); // Keep incomplete event in buffer
                    
                    for (const event of events) {
                        if (!event.trim()) continue;
                        // Extract JSON from "data: {...}"
                        const match = event.match(/^data:\\s*(.+)$/s);
                        if (!match) continue;
                        try {
                            const data = JSON.parse(match[1]);
                            
                            if (data.type === 'timing') {
                                timing[data.stage] = data.ms;
                                updateTiming();
                            } else if (data.type === 'memories') {
                                updateMemories(data.memories, [], []);
                            } else if (data.type === 'memory_update') {
                                memoryChanges++;
                                updateMemories(data.memories, data.added || [], data.removed || []);
                                updateTiming();
                            } else if (data.type === 'thinking') {
                                if (!firstToken) { firstToken = true; timing.firstToken = Date.now() - startTime; updateTiming(); }
                                thinkingPanel.style.display = 'block';
                                fullThinking += data.t;
                                thinkingPanel.querySelector('.thinking-content').textContent = fullThinking;
                            } else if (data.type === 'token') {
                                if (!firstToken) { firstToken = true; timing.firstToken = Date.now() - startTime; updateTiming(); }
                                fullResponse += data.t;
                                timing.tokens = (timing.tokens || 0) + 1;
                                assistantMsg.innerHTML = formatMarkdown(fullResponse) + '<span class="cursor"></span>';
                                chat.scrollTop = chat.scrollHeight;
                            } else if (data.type === 'timeline') {
                                createScrubber(responseContainer, data.data);
                            } else if (data.type === 'done') {
                                timingPanel.innerHTML += `<span>total: ${Date.now() - startTime}ms</span>`;
                                assistantMsg.innerHTML = formatMarkdown(fullResponse);
                                history.push({role: 'user', content: msg}, {role: 'assistant', content: fullResponse});
                            }
                        } catch (e) { console.log('Parse error:', e, line); }
                    }
                }
            } catch (e) { assistantMsg.innerHTML = `<span style="color:#f85149;">Error: ${e.message}</span>`; }
            
            sendBtn.disabled = false;
            input.focus();
            
            function updateTiming() {
                const parts = [];
                if (timing.init) parts.push(`<span class="${timing.init > 5000 ? 'slow' : timing.init > 1000 ? 'medium' : 'fast'}">init: ${timing.init}ms</span>`);
                if (timing.embed) parts.push(`<span class="${timing.embed > 2000 ? 'slow' : timing.embed > 500 ? 'medium' : 'fast'}">embed: ${timing.embed}ms</span>`);
                if (timing.firstToken) parts.push(`<span class="${timing.firstToken > 5000 ? 'slow' : timing.firstToken > 2000 ? 'medium' : 'fast'}">first token: ${timing.firstToken}ms</span>`);
                if (timing.tokens) parts.push(`<span>tokens: ${timing.tokens}</span>`);
                if (memoryChanges) parts.push(`<span class="fast">🔄 updates: ${memoryChanges}</span>`);
                timingPanel.innerHTML = '⏱️ ' + parts.join('');
            }
            
            function updateMemories(memories, added, removed) {
                let html = `<div class="memories-header">📚 Active memories (${memories.length})</div>`;
                memories.forEach(m => { html += `<div class="memory-item ${added.includes(m) ? 'new' : ''}">${escapeHtml(m)}</div>`; });
                removed.forEach(m => { html += `<div class="memory-item removed">${escapeHtml(m)}</div>`; });
                memoriesPanel.innerHTML = html;
            }
        }
        
        function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
        function formatMarkdown(t) { return t.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>').replace(/\\*(.+?)\\*/g, '<em>$1</em>').replace(/`(.+?)`/g, '<code style="background:#21262d;padding:2px 6px;border-radius:3px;">$1</code>').replace(/\\n/g, '<br>'); }
        
        function createScrubber(container, timeline) {
            if (!timeline?.length) return;
            const scrubber = document.createElement('div');
            scrubber.className = 'scrubber-panel';
            let idx = timeline.length - 1, playing = false, interval;
            
            function render() {
                const item = timeline[idx];
                const before = timeline.slice(0, idx + 1);
                scrubber.innerHTML = `
                    <div class="scrubber-header">🔄 Timeline (${timeline.length} tokens)</div>
                    <div class="playback-controls"><button id="rwdBtn">⏮</button><button id="playBtn">${playing ? '⏸' : '▶'}</button><button id="fwdBtn">⏭</button></div>
                    <div class="scrubber-info"><span>Token ${idx + 1}/${timeline.length}</span><span>${item.type}</span></div>
                    <input type="range" class="scrubber-slider" min="0" max="${timeline.length - 1}" value="${idx}">
                    <div class="scrubber-token">${before.slice(-15).map((t,i) => i === before.slice(-15).length - 1 ? `<span class="current">${escapeHtml(t.token)}</span>` : escapeHtml(t.token)).join('')}</div>
                    <div class="scrubber-memories"><div style="color:#f0883e;margin-bottom:4px;">Memories at token:</div>${(item.memories||[]).map(m => `<div class="memory-item">${escapeHtml(m.substring(0,80))}...</div>`).join('')}</div>
                `;
                scrubber.querySelector('.scrubber-slider').oninput = e => { idx = +e.target.value; render(); };
                scrubber.querySelector('#rwdBtn').onclick = () => { idx = 0; render(); };
                scrubber.querySelector('#fwdBtn').onclick = () => { idx = timeline.length - 1; render(); };
                scrubber.querySelector('#playBtn').onclick = () => {
                    playing = !playing;
                    if (playing) interval = setInterval(() => { if (idx < timeline.length - 1) { idx++; render(); } else { playing = false; clearInterval(interval); render(); } }, 100);
                    else clearInterval(interval);
                    render();
                };
            }
            render();
            container.appendChild(scrubber);
        }
    </script>
</body>
</html>
"""


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("openai-secret")],
)
class TutorService:
    
    @modal.enter()
    def startup(self):
        """Runs once when container starts - loads model and pre-embeds all memories."""
        import sys
        sys.path.insert(0, "/app")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from openai import OpenAI
        from streaming_memory import MemoryPool
        
        print(f"🚀 Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"✅ Model loaded! Device: {next(self.model.parameters()).device}")
        
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.embed_cache = {}
        
        def embed(text: str):
            if text in self.embed_cache:
                return self.embed_cache[text]
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
            )
            result = response.data[0].embedding
            self.embed_cache[text] = result
            return result
        
        self.embed_fn = embed
        
        # Create a pool for each scenario
        self.pools = {}
        self.pool_total_tokens = {}  # Total tokens if ALL memories were included
        
        for scenario_name, scenario_config in SCENARIOS.items():
            print(f"📚 Loading {scenario_name} memories...")
            
            pool = MemoryPool(
                embed_fn=embed,
                softmax_temperature=0.15,
                diversity_weight=0.5,
                association_weight=0.5,
            )
            
            with open(scenario_config["memory_file"]) as f:
                memories = json.load(f)
            
            # Batch embed all memories
            memory_contents = [m["content"] for m in memories]
            batch_size = 50
            for i in range(0, len(memory_contents), batch_size):
                batch = memory_contents[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                for j, emb_data in enumerate(response.data):
                    self.embed_cache[batch[j]] = emb_data.embedding
            
            # Add memories to pool
            for mem in memories:
                created_str = mem.get("created_at", "")
                try:
                    dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    created_at = dt.replace(tzinfo=None)
                except:
                    created_at = datetime.now()
                
                pool.add(
                    content=mem["content"],
                    emotional_intensity=mem.get("emotional_intensity", 0.5),
                    created_at=created_at,
                )
            
            self.pools[scenario_name] = pool
            
            # Calculate total tokens for ALL memories in this pool
            all_memory_text = "\n".join([f"- {m['content']}" for m in memories])
            total_pool_tokens = len(self.tokenizer.encode(all_memory_text))
            self.pool_total_tokens[scenario_name] = total_pool_tokens
            
            print(f"  ✅ Loaded {len(memories)} {scenario_name} memories ({total_pool_tokens} tokens)")
        
        print("🟢 Container ready - responses will be fast!")
    
    @modal.asgi_app()
    def serve(self):
        import torch
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        
        web_app = FastAPI()
        
        # Allow CORS for React frontend
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        def format_memories(memories, prefix: str) -> str:
            if not memories:
                return ""
            lines = [prefix]
            for mem in memories:
                lines.append(f"- {mem.content}")
            return "\n".join(lines)
        
        @web_app.get("/", response_class=HTMLResponse)
        async def home():
            return HTML_PAGE
        
        @web_app.get("/health")
        async def health():
            """Health check endpoint for warming up the container."""
            return {"status": "ok", "model": MODEL_ID}
        
        @web_app.post("/chat/stream")
        async def chat_stream(request: Request):
            import time
            
            data = await request.json()
            message = data.get("message", "")
            history = data.get("history", [])
            update_every_n = data.get("update_every_n", 1)
            max_memories = data.get("max_memories", 5)
            lookback_tokens = data.get("lookback_tokens", 60)
            scenario_name = data.get("scenario", "tutor")
            
            # Get scenario config
            scenario = SCENARIOS.get(scenario_name, SCENARIOS["tutor"])
            system_prompt = scenario["system_prompt"]
            memory_prefix = scenario["memory_prefix"]
            
            model = self.model
            tokenizer = self.tokenizer
            pool = self.pools[scenario_name]
            all_memories_tokens = self.pool_total_tokens[scenario_name]
            
            def generate():
                # SSE format: "data: {...}\n\n"
                def sse(data):
                    return f"data: {json.dumps(data)}\n\n"
                
                t0 = time.time()
                yield sse({'type': 'timing', 'stage': 'init', 'ms': 0})
                
                # Build query
                query = message
                if history:
                    recent = " ".join([m["content"] for m in history[-4:]])
                    query = recent + " " + message
                
                # Initial retrieval
                t1 = time.time()
                memories = pool.retrieve(query, max_memories=max_memories)
                yield sse({'type': 'timing', 'stage': 'embed', 'ms': int((time.time() - t1) * 1000)})
                
                current_mem_contents = [m.content for m in memories]
                yield sse({'type': 'memories', 'memories': current_mem_contents})
                
                # Track ALL unique memories seen and their token counts
                all_unique_memories = set(current_mem_contents)
                memory_token_cache = {}  # cache token counts per memory
                
                def get_memory_tokens(mem_content):
                    if mem_content not in memory_token_cache:
                        memory_token_cache[mem_content] = len(tokenizer.encode(mem_content))
                    return memory_token_cache[mem_content]
                
                # Calculate initial memory tokens
                for mem in current_mem_contents:
                    get_memory_tokens(mem)
                
                # First, calculate BASE context (without memories)
                base_messages = [
                    {"role": "system", "content": system_prompt},
                ]
                for h in history[-6:]:
                    base_messages.append(h)
                base_messages.append({"role": "user", "content": message})
                base_text = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
                base_context_size = len(tokenizer.encode(base_text))
                
                # Build full prompt with memories
                memory_context = format_memories(memories, memory_prefix)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": memory_context},
                ]
                for h in history[-6:]:
                    messages.append(h)
                messages.append({"role": "user", "content": message})
                
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
                
                max_tokens = 1500
                # update_every_n comes from request
                
                all_tokens = []
                in_thinking = False
                timeline = []
                token_idx = 0
                token_history = []  # Track token counts over time for graph
                
                # Track initial context size
                current_memory_tokens = sum(get_memory_tokens(m) for m in current_mem_contents)
                rag_memory_tokens = sum(get_memory_tokens(m) for m in all_unique_memories)  # unique seen so far
                
                token_history.append({
                    'token_idx': 0,
                    'streaming': base_context_size + current_memory_tokens,
                    'rag': base_context_size + rag_memory_tokens,
                    'all': base_context_size + all_memories_tokens,
                })
                
                yield sse({
                    'type': 'context_size', 
                    'base_context_size': base_context_size,
                    'current_memory_tokens': current_memory_tokens,
                    'rag_memory_tokens': rag_memory_tokens,
                    'all_memories_tokens': all_memories_tokens,
                    'unique_memories': len(all_unique_memories),
                    'token_history': token_history,
                })
                
                with torch.no_grad():
                    current_ids = input_ids
                    
                    while len(all_tokens) < max_tokens:
                        chunk_size = min(update_every_n, max_tokens - len(all_tokens))
                        
                        outputs = model.generate(
                            current_ids,
                            max_new_tokens=chunk_size,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        
                        new_token_ids = outputs[0, current_ids.shape[1]:].tolist()
                        
                        for tid in new_token_ids:
                            all_tokens.append(tid)
                            token_text = tokenizer.decode([tid], skip_special_tokens=False)
                            
                            if '<think>' in token_text:
                                in_thinking = True
                                continue
                            elif '</think>' in token_text:
                                in_thinking = False
                                continue
                            
                            if in_thinking:
                                timeline.append({'idx': token_idx, 'token': token_text, 'type': 'thinking', 'memories': current_mem_contents.copy()})
                                token_idx += 1
                                yield sse({'type': 'thinking', 't': token_text})
                            else:
                                clean_token = tokenizer.decode([tid], skip_special_tokens=True)
                                if clean_token:
                                    timeline.append({'idx': token_idx, 'token': clean_token, 'type': 'response', 'memories': current_mem_contents.copy()})
                                    token_idx += 1
                                    yield sse({'type': 'token', 't': clean_token})
                            
                            if tid == tokenizer.eos_token_id:
                                break
                        
                        if tokenizer.eos_token_id in new_token_ids:
                            break
                        
                        # Re-retrieve memories using ONLY the recent generated tokens
                        # This allows the model's reasoning to drive memory retrieval
                        lookback_text = tokenizer.decode(all_tokens[-lookback_tokens:], skip_special_tokens=True)
                        
                        new_memories = pool.retrieve(lookback_text, max_memories=max_memories)
                        new_mem_contents = [m.content for m in new_memories]
                        
                        if set(new_mem_contents) != set(current_mem_contents):
                            added = [m for m in new_mem_contents if m not in current_mem_contents]
                            removed = [m for m in current_mem_contents if m not in new_mem_contents]
                            
                            # Track all unique memories seen
                            for mem in new_mem_contents:
                                all_unique_memories.add(mem)
                                get_memory_tokens(mem)  # cache token count
                            
                            # Raw token concatenation fix
                            current_mem_contents = new_mem_contents
                            memory_context = format_memories(new_memories, memory_prefix)
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "system", "content": memory_context},
                            ]
                            for h in history[-6:]:
                                messages.append(h)
                            messages.append({"role": "user", "content": message})
                            
                            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            new_prefix_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
                            
                            generated_tensor = torch.tensor([all_tokens], device=model.device)
                            current_ids = torch.cat([new_prefix_ids, generated_tensor], dim=1)
                            
                            # Track context size after memory swap
                            current_memory_tokens = sum(get_memory_tokens(m) for m in current_mem_contents)
                            rag_memory_tokens = sum(get_memory_tokens(m) for m in all_unique_memories)
                            
                            token_history.append({
                                'token_idx': len(all_tokens),
                                'streaming': base_context_size + current_memory_tokens,
                                'rag': base_context_size + rag_memory_tokens,
                                'all': base_context_size + all_memories_tokens,
                            })
                            
                            yield sse({
                                'type': 'memory_update',
                                'memories': new_mem_contents,
                                'added': added,
                                'removed': removed,
                                'base_context_size': base_context_size,
                                'current_memory_tokens': current_memory_tokens,
                                'rag_memory_tokens': rag_memory_tokens,
                                'all_memories_tokens': all_memories_tokens,
                                'unique_memories': len(all_unique_memories),
                                'token_history': token_history,
                            })
                        else:
                            current_ids = outputs
                
                # Check if we hit max tokens without EOS
                hit_max = len(all_tokens) >= max_tokens and tokenizer.eos_token_id not in all_tokens[-10:]
                
                yield sse({'type': 'timeline', 'data': timeline})
                if hit_max:
                    yield sse({'type': 'max_tokens', 'limit': max_tokens})
                yield sse({'type': 'done'})
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",  # Critical for real-time streaming
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        
        return web_app
