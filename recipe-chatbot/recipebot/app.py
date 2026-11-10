"""Web UI with VOICE input/output.   Run:  python -m recipebot.app      then open http://127.0.0.1:5000  (Chrome/Edge)
Voice uses the browser's Web Speech API (microphone -> text, text -> speech): no extra server, no extra API key.
Each browser tab gets its own conversation state (Chat); the dataset and index are loaded once and shared.
Deploy:  gunicorn -w 1 --threads 4 -t 180 "recipebot.app:create_app()"
Env: GEMINI_API_KEY, GEMINI_MODEL (optional; without a key the rule-based router is used), RECIPEBOT_SUMMARY=0 to disable LLM summaries."""
import os
import threading
import uuid
from collections import OrderedDict

from flask import Flask, jsonify, request

from .chat import Chat
from .data import load_recipes
from .llm import Gemini
from .search import SearchIndex

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recipe Assistant</title><style>
:root{--bg:#faf7f2;--card:#fff;--ink:#222;--mut:#6b6b6b;--acc:#c2410c;--line:#e7e0d6}
@media (prefers-color-scheme:dark){:root{--bg:#1b1a18;--card:#262421;--ink:#f1ece4;--mut:#a9a299;--acc:#fb923c;--line:#3a3631}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;flex:1}label{color:var(--mut);font-size:13px}select,button,input{font:inherit}
#log{flex:1;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.m{max-width:min(860px,94%);padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word;border:1px solid var(--line);background:var(--card)}
.u{align-self:flex-end;background:var(--acc);color:#fff;border-color:var(--acc)}.meta{font-size:11px;color:var(--mut);margin-top:4px}
footer{padding:12px 16px;border-top:1px solid var(--line);display:flex;gap:8px}
#q{flex:1;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
button{padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
#mic.on{background:#dc2626;color:#fff;border-color:#dc2626}#send{background:var(--acc);color:#fff;border-color:var(--acc)}
.hint{color:var(--mut);font-size:13px;padding:0 16px 8px}
</style></head><body>
<header><h1>Recipe Assistant <span style="font-weight:400;color:var(--mut)">grounded in the Recipe Box dataset</span></h1>
<label><input type="checkbox" id="speak" checked> Speak replies</label>
<label>Voice <select id="lang"><option value="en-US">English (US)</option><option value="en-IN">English (India)</option><option value="en-GB">English (UK)</option></select></label>
<button id="reset" title="Clear search and restrictions">Reset</button></header>
<div id="log"></div>
<div class="hint">Try: "find a penne pasta recipe", then "I cannot have dairy" &middot; "I have eggs, onions, tomatoes and rice, at most 6 steps" &middot; "remove eggs from recipe 1"</div>
<footer><button id="mic" title="Hold a conversation by voice">&#127908; Speak</button><input id="q" placeholder="Type or press the microphone..." autocomplete="off"><button id="send">Send</button></footer>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),mic=document.getElementById('mic');
const sid=localStorage.sid||(localStorage.sid=(crypto.randomUUID?crypto.randomUUID():String(Math.random()).slice(2)));
function add(t,cls,meta){const d=document.createElement('div');d.className='m '+cls;d.textContent=t;if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;d.appendChild(m)}log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
function speak(t){if(!document.getElementById('speak').checked||!window.speechSynthesis||!t)return;speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(t.replace(/\[[^\]]*\]/g,'').replace(/[*_`#]/g,''));u.lang=document.getElementById('lang').value;speechSynthesis.speak(u)}
async function send(text){text=(text||q.value).trim();if(!text)return;q.value='';add(text,'u');const w=add('...','');
  try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,sid})});
  const j=await r.json();w.remove();if(j.error){add('Error: '+j.error,'')}else{add(j.text,'',`router: ${j.router} · tool: ${j.tool}`+(j.llm_notice?` · ${j.llm_notice}`:''));speak(j.speech)}}
  catch(e){w.remove();add('Network error: '+e,'')}}
document.getElementById('send').onclick=()=>send();q.onkeydown=e=>{if(e.key==='Enter')send()};
document.getElementById('reset').onclick=()=>send('reset');
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;let rec=null;
if(!SR){mic.disabled=true;mic.title='Speech recognition is not supported in this browser (use Chrome or Edge).'}
else{mic.onclick=()=>{if(rec){rec.stop();return}if(window.speechSynthesis)speechSynthesis.cancel();
  rec=new SR();rec.lang=document.getElementById('lang').value;rec.interimResults=true;rec.maxAlternatives=1;let final='';
  rec.onstart=()=>mic.classList.add('on');
  rec.onresult=e=>{let t='';for(const r of e.results){t+=r[0].transcript;if(r.isFinal)final=t}q.value=t};
  rec.onerror=e=>{add('Microphone: '+e.error+(e.error==='not-allowed'?' (allow microphone access in the browser)':''),'')};
  rec.onend=()=>{mic.classList.remove('on');rec=null;const t=(final||q.value).trim();if(t)send(t)};rec.start()}}
add("Hi! Ask me for a recipe by typing or using the microphone. Everything I say comes from the recipe dataset; if it can't support an answer, I'll say so.",'');
</script></body></html>"""


def create_app(df=None, index=None, llm="auto"):
    df = df if df is not None else load_recipes()
    index = index or SearchIndex.load_or_build(df, semantic=True, build_semantic=False)
    if llm == "auto":
        g = Gemini()
        llm = g if g.available else None
    summary = os.environ.get("RECIPEBOT_SUMMARY", "1") != "0"
    sessions, lock = OrderedDict(), threading.Lock()
    app = Flask(__name__)
    if index.semantic_notice:
        app.logger.warning(index.semantic_notice)

    def chat_for(sid):
        with lock:
            if sid not in sessions:
                sessions[sid] = Chat(df, index, llm=llm, summary=summary)
                while len(sessions) > 200:
                    sessions.popitem(last=False)
            sessions.move_to_end(sid)
            return sessions[sid]

    @app.get("/")
    def home():
        return PAGE

    @app.get("/health")
    def health():
        return jsonify(ok=True, recipes=len(df), llm=bool(llm), model=getattr(llm, "model", None))

    @app.post("/api/chat")
    def api_chat():
        body = request.get_json(silent=True) or {}
        msg = (body.get("message") or "").strip()
        if not msg:
            return jsonify(error="Empty message"), 400
        chat = chat_for((body.get("sid") or str(uuid.uuid4()))[:64])
        try:
            with lock:  # engines share numpy/pandas state; keep turns serialized per process
                turn = chat.handle(msg[:1000])
                visible = {
                    key: turn[key]
                    for key in ("text", "summary", "speech", "tool", "action", "router", "llm_notice")
                }
                return jsonify(visible)
        except Exception as e:  # never leak a traceback to the browser
            app.logger.exception("chat failure")
            return jsonify(error=f"Internal error ({type(e).__name__}). Try rephrasing or press Reset."), 500

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))