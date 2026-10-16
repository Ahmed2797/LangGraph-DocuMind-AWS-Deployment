'use strict';
/* ============================================================
   RagVision frontend
   ------------------------------------------------------------
   Backend contract (FastAPI):

   POST /api/chat   (multipart/form-data)
     fields: query, chat_id, files[] (any), images[] (image), context_image (optional url)
     response: NDJSON stream, one JSON per line:
       {"type":"node","name":"decide_retrieval","status":"start"}   // status: start | end
       {"type":"node","name":"decide_retrieval","status":"end"}
       {"type":"token","text":"partial answer "}
       {"type":"final","answer":"full text",
        "media":[{"type":"image","url":"https://..","caption":"..","source":"https://.."},
                 {"type":"chart","url":"https://.."},
                 {"type":"table","columns":["A","B"],"rows":[["1","2"]]}],
        "sources":[{"title":"..","url":"https://.."}]}

   POST /api/transcribe (multipart: audio)  ->  {"text":"..."}

   Backend not found MOCK_FALLBACK=true demo mode run
   ============================================================ */

const CONFIG = { API_BASE: '', MOCK_FALLBACK: true };

/* ---------- helpers ---------- */
const $ = (s, r = document) => r.querySelector(s);
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('show'), 2600); }

/* offline fallback: small LaTeX → HTML converter (used when KaTeX is not loaded) */
const TEX_SYM = {
  alpha:'α',beta:'β',gamma:'γ',delta:'δ',epsilon:'ε',theta:'θ',lambda:'λ',mu:'μ',pi:'π',sigma:'σ',phi:'φ',omega:'ω',
  Sigma:'Σ',Delta:'Δ',Omega:'Ω',times:'×',cdot:'·',in:'∈',leq:'≤',le:'≤',geq:'≥',ge:'≥',neq:'≠',approx:'≈',infty:'∞',
  sum:'∑',prod:'∏',rightarrow:'→',to:'→',leftarrow:'←',partial:'∂',nabla:'∇',pm:'±',ldots:'…',dots:'…',top:'ᵀ',
};
function texLite(src) {
  const s = String(src), MB = { R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', C: 'ℂ' };
  let i = 0;
  const seq = end => { let h = ''; while (i < s.length && s[i] !== end) h += item(); return h; };
  const arg = () => {
    while (s[i] === ' ') i++;
    if (s[i] === '{') { i++; const h = seq('}'); i++; return h; }
    if (s[i] === '\\') return command();
    return item();
  };
  function command() {
    i++;
    if (!/[A-Za-z]/.test(s[i] || '')) { const c = s[i++] || ''; return /[,;: ]/.test(c) ? ' ' : c === '\\' ? '<br>' : esc(c); }
    let name = ''; while (/[A-Za-z]/.test(s[i] || '')) name += s[i++];
    switch (name) {
      case 'frac': case 'dfrac': { const a = arg(), b = arg(); return `<span class="tl-frac"><span>${a}</span><span>${b}</span></span>`; }
      case 'sqrt': return `<span class="tl-sqrt">√<span>${arg()}</span></span>`;
      case 'text': case 'mathrm': case 'mathbf': case 'mathit': case 'textbf': case 'operatorname': case 'mathcal':
        return `<span class="tl-rm">${arg()}</span>`;
      case 'mathbb': { const a = arg().replace(/<[^>]+>/g, ''); return MB[a] || a; }
      case 'left': case 'right': case 'big': case 'Big': case 'bigg': return '';
      case 'quad': case 'qquad': return '&emsp;';
      default: return TEX_SYM[name] ?? `<span class="tl-rm">${esc(name)}</span>`;   // softmax, max, log …
    }
  }
  function item() {
    const ch = s[i];
    if (ch === '{') { i++; const h = seq('}'); i++; return h; }
    if (ch === '\\') return command();
    if (ch === '^') { i++; return `<sup>${arg()}</sup>`; }
    if (ch === '_') { i++; return `<sub>${arg()}</sub>`; }
    i++;
    return /[A-Za-z]/.test(ch) ? `<i>${ch}</i>` : esc(ch);
  }
  let out = ''; while (i < s.length) out += item();
  return out;
}

/* markdown renderer: headings, bold/italic, links, code, tables, nested/numbered lists, LaTeX (KaTeX) */
function renderMath(tex, display) {
  tex = tex.trim();
  if (window.katex) { try { return window.katex.renderToString(tex, { displayMode: display, throwOnError: false }); } catch { /* fall through */ } }
  return display ? `<div class="tl tl-display">${texLite(tex)}</div>` : `<span class="tl">${texLite(tex)}</span>`;
}

function md(src) {
  const stash = [];
  const keep = html => `\u0000${stash.push(html) - 1}\u0000`;
  const inline = s => s
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*\n]*)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  let t = String(src ?? '');
  t = t.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, l, c) => keep(`<pre><code>${esc(c.replace(/\n$/, ''))}</code></pre>`));
  t = t.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => keep(renderMath(m, true)))
       .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => keep(renderMath(m, true)))
       .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => keep(renderMath(m, false)));
  t = t.replace(/`([^`\n]+)`/g, (_, c) => keep(`<code>${esc(c)}</code>`));
  t = esc(t);

  const lines = t.split('\n');
  let out = '', stack = [];                       // open lists: {indent, tag}
  const closeLists = () => { while (stack.length) out += `</li></${stack.pop().tag}>`; };
  const isHolder = l => /^\u0000\d+\u0000$/.test(l.trim());

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;                   // blank lines never close a list

    const li = line.match(/^(\s*)([-*•]|\d+[.)])\s+(.*)$/);
    if (li) {
      const indent = li[1].length, ordered = /\d/.test(li[2]), tag = ordered ? 'ol' : 'ul';
      while (stack.length && indent < stack[stack.length - 1].indent) out += `</li></${stack.pop().tag}>`;
      const top = stack[stack.length - 1];
      if (!top || indent > top.indent) {
        const n = parseInt(li[2], 10);
        out += `<${tag}${ordered && n > 1 ? ` start="${n}"` : ''}><li>${inline(li[3])}`;   // numbering survives a split list
        stack.push({ indent, tag });
      } else out += `</li><li>${inline(li[3])}`;
      continue;
    }
    // text or a formula/code block that belongs to the open list item (indented)
    if (stack.length && (/^\s+\S/.test(line) || isHolder(line))) {
      out += isHolder(line) ? line.trim() : `<div>${inline(line.trim())}</div>`;
      continue;
    }
    closeLists();

    if (/^\s*\|/.test(line)) {                    // table
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i].trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim())); i++; }
      i--;
      const body = rows.filter(r => !r.every(c => /^:?-{2,}:?$/.test(c)));
      if (body.length) {
        const th = body[0].map(c => `<th>${inline(c)}</th>`).join('');
        const tr = body.slice(1).map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('');
        out += `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
      }
      continue;
    }
    let m;
    if ((m = line.match(/^\s*(#{1,4})\s+(.*)$/))) out += `<h4>${inline(m[2])}</h4>`;
    else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) out += '<hr>';
    else if ((m = line.match(/^\s*&gt;\s?(.*)$/))) out += `<blockquote>${inline(m[1])}</blockquote>`;
    else if (isHolder(line)) out += line.trim();
    else out += `<p>${inline(line.trim())}</p>`;
  }
  closeLists();
  return out.replace(/\u0000(\d+)\u0000/g, (_, k) => stash[k]);
}

/* ---------- state + persistence ---------- */
const STORE_KEY = 'ragvision_v1';
let state = (() => { try { return JSON.parse(localStorage.getItem(STORE_KEY)); } catch { return null; } })()
  || { projects: [], chats: [], activeId: null };

function save() {
  // blob: URLs die after reload, so don't persist them
  const json = JSON.stringify(state, (k, v) => (k === 'url' && typeof v === 'string' && v.startsWith('blob:')) ? undefined : v);
  try { localStorage.setItem(STORE_KEY, json); } catch { toast('Storage full — old chat delete'); }
}
const activeChat = () => state.chats.find(c => c.id === state.activeId);

/* ---------- dialogs & menus ---------- */
function openDialog({ title, message = '', input = null, options = null, okText = 'OK', danger = false }) {
  return new Promise(resolve => {
    const d = $('#dlg');
    $('#dlgTitle').textContent = title;
    $('#dlgMsg').textContent = message;
    const inp = $('#dlgInput'), sel = $('#dlgSelect');
    inp.style.display = input !== null ? '' : 'none';
    sel.style.display = options ? '' : 'none';
    if (input !== null) inp.value = input;
    if (options) sel.innerHTML = options.map(o => `<option value="${esc(o.value)}">${esc(o.label)}</option>`).join('');
    const ok = $('#dlgOk'); ok.textContent = okText; ok.className = 'btn ' + (danger ? 'danger' : 'primary');
    let result = null;
    const done = () => { d.close(); d.removeEventListener('close', onClose); resolve(result); };
    const onClose = () => resolve(result);
    $('#dlgForm').onsubmit = e => { e.preventDefault(); result = options ? sel.value : (input !== null ? inp.value.trim() : true); if (input !== null && !result) return; done(); };
    $('#dlgCancel').onclick = () => { result = null; done(); };
    d.addEventListener('close', onClose);
    d.showModal();
    if (input !== null) { inp.focus(); inp.select(); }
  });
}

function showMenu(anchor, items) {
  document.querySelectorAll('.menu').forEach(m => m.remove());
  const m = el('div', 'menu');
  items.forEach(it => {
    const b = el('button', it.danger ? 'danger' : '', esc(it.label));
    b.onclick = () => { m.remove(); it.fn(); };
    m.append(b);
  });
  document.body.append(m);
  const r = anchor.getBoundingClientRect();
  m.style.top = Math.min(r.bottom + 4, innerHeight - m.offsetHeight - 8) + 'px';
  m.style.left = Math.min(r.left, innerWidth - m.offsetWidth - 8) + 'px';
  setTimeout(() => document.addEventListener('click', () => m.remove(), { once: true }), 0);
}

/* ---------- chats & projects ---------- */
function newChat(projectId = null) {
  const c = { id: uid(), title: 'New chat', projectId, messages: [], createdAt: Date.now(), updatedAt: Date.now() };
  state.chats.unshift(c); state.activeId = c.id;
  if (projectId) { const p = state.projects.find(p => p.id === projectId); if (p) p.open = true; }
  save(); renderAll(); $('#input').focus(); closeMobile();
}
function setActive(id) { state.activeId = id; save(); renderAll(); closeMobile(); }

async function renameChat(c) {
  const name = await openDialog({ title: 'Rename chat', input: c.title, okText: 'Save' });
  if (name) { c.title = name; save(); renderAll(); }
}
async function deleteChat(c) {
  if (!await openDialog({ title: 'Delete chat?', message: `"${c.title}" permanently delete`, okText: 'Delete', danger: true })) return;
  state.chats = state.chats.filter(x => x.id !== c.id);
  if (state.activeId === c.id) state.activeId = state.chats[0]?.id ?? null;
  if (!state.activeId) return newChat();
  save(); renderAll();
}
async function moveChat(c) {
  const opts = [{ value: '', label: '— No project —' }, ...state.projects.map(p => ({ value: p.id, label: p.name }))];
  const v = await openDialog({ title: 'Move to project', options: opts, okText: 'Move' });
  if (v === null) return;
  c.projectId = v || null; save(); renderAll();
}
async function newProject() {
  const name = await openDialog({ title: 'New project', input: '', okText: 'Create' });
  if (name) { state.projects.push({ id: uid(), name, open: true }); save(); renderSidebar(); }
}
async function renameProject(p) {
  const name = await openDialog({ title: 'Rename project', input: p.name, okText: 'Save' });
  if (name) { p.name = name; save(); renderSidebar(); }
}
async function deleteProject(p) {
  if (!await openDialog({ title: 'Delete project?', message: `"${p.name}" delete. Inside chat ungrouped`, okText: 'Delete', danger: true })) return;
  state.chats.forEach(c => { if (c.projectId === p.id) c.projectId = null; });
  state.projects = state.projects.filter(x => x.id !== p.id);
  save(); renderSidebar();
}

/* ---------- rendering: sidebar ---------- */
function chatItem(c, nested) {
  const d = el('div', `chat-item${nested ? ' nested' : ''}${c.id === state.activeId ? ' active' : ''}`,
    `<span>💬</span><span class="ctitle">${esc(c.title)}</span><button class="kebab" title="Options">⋯</button>`);
  d.onclick = e => { if (!e.target.closest('.kebab')) setActive(c.id); };
  d.querySelector('.kebab').onclick = e => {
    e.stopPropagation();
    showMenu(e.currentTarget, [
      { label: '✎ Rename', fn: () => renameChat(c) },
      { label: '📁 Move to project', fn: () => moveChat(c) },
      { label: '🗑 Delete', danger: true, fn: () => deleteChat(c) },
    ]);
  };
  return d;
}

function renderSidebar() {
  const q = $('#searchInput').value.toLowerCase().trim();
  const list = $('#sideList'); list.innerHTML = '';
  const match = c => !q || c.title.toLowerCase().includes(q);
  const sorted = [...state.chats].sort((a, b) => b.updatedAt - a.updatedAt);

  state.projects.forEach(p => {
    const chats = sorted.filter(c => c.projectId === p.id && match(c));
    if (q && !chats.length) return;
    const head = el('div', 'group-head',
      `<span>${p.open ? '▾' : '▸'}</span><span class="gname">📁 ${esc(p.name)}</span><span class="count">${chats.length}</span><button class="kebab" title="Options">⋯</button>`);
    head.onclick = e => { if (e.target.closest('.kebab')) return; p.open = !p.open; save(); renderSidebar(); };
    head.querySelector('.kebab').onclick = e => {
      e.stopPropagation();
      showMenu(e.currentTarget, [
        { label: '＋ New chat here', fn: () => newChat(p.id) },
        { label: '✎ Rename', fn: () => renameProject(p) },
        { label: '🗑 Delete project', danger: true, fn: () => deleteProject(p) },
      ]);
    };
    list.append(head);
    if (p.open || q) chats.forEach(c => list.append(chatItem(c, true)));
  });

  const loose = sorted.filter(c => !c.projectId && match(c));
  if (loose.length) { list.append(el('div', 'section-label', 'Chats')); loose.forEach(c => list.append(chatItem(c, false))); }
  if (!list.children.length) list.append(el('div', 'empty', 'Please, Create New Chat'));
}

/* ---------- rendering: messages ---------- */
function welcome() {
  const w = el('div', 'welcome', `<h2>◉ DocuMind AI Agent</h2><p> Text, image, file and voice to asked query। On the right side the workflow graph lights up node by node</p>
    <div class="pills"><span class="pill">Explain this chart</span><span class="pill">Summarize my PDF</span><span class="pill">Search the web with images</span></div>`);
  w.querySelectorAll('.pill').forEach(p => p.onclick = () => { $('#input').value = p.textContent; $('#input').focus(); });
  return w;
}

function mediaEl(m) {
  if (m.type === 'table') {
    const w = el('div', 'table-wrap');
    const head = (m.columns || []).map(c => `<th>${esc(c)}</th>`).join('');
    const rows = (m.rows || []).map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join('')}</tr>`).join('');
    w.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
    return w;
  }
  const f = el('figure');
  f.innerHTML = `<img loading="lazy" src="${esc(m.url)}" alt="${esc(m.caption || 'image')}"><figcaption>${esc(m.caption || m.type)}</figcaption>`;
  f.onclick = () => openDrawer(m);
  return f;
}

function msgEl(m) {
  const w = el('div', `msg ${m.role}`);
  w.append(el('div', 'avatar', m.role === 'user' ? '🙂' : '◉'));
  const body = el('div', 'body'); w.append(body);

  if (m.attachments?.length) {
    const row = el('div', 'att-row');
    m.attachments.forEach(a => {
      if (a.kind === 'image' && a.url) {
        const im = el('img', 'att-img'); im.src = a.url; im.onclick = () => openDrawer({ url: a.url, caption: a.name });
        row.append(im);
      } else row.append(el('div', 'chip', `<span>${a.kind === 'image' ? '🖼️' : '📄'} ${esc(a.name)}</span>`));
    });
    body.append(row);
  }

  const bubble = el('div', 'bubble');
  if (m.pending && !m.text) bubble.innerHTML = '<div class="typing"><i></i><i></i><i></i></div>';
  else bubble.innerHTML = m.role === 'user' ? `<p>${esc(m.text).replace(/\n/g, '<br>')}</p>` : md(m.text);
  body.append(bubble);

  if (m.media?.length) { const g = el('div', 'media-grid'); m.media.forEach(x => g.append(mediaEl(x))); body.append(g); }
  if (m.sources?.length) {
    const s = el('div', 'sources');
    m.sources.forEach(x => s.insertAdjacentHTML('beforeend', /^https?:/.test(x.url || '')
      ? `<a href="${esc(x.url)}" target="_blank" rel="noopener" title="${esc(x.title || x.url)}">🔗 ${esc(x.title || x.url)}</a>`
      : `<span class="src" title="${esc(x.title)}">📄 ${esc(x.title)}</span>`));
    body.append(s);
  }
  if (m.role === 'assistant' && !m.pending && m.text) {
    const a = el('div', 'msg-actions', '<button data-a="listen">🔊 Listen</button><button data-a="copy">⧉ Copy</button>');
    a.querySelector('[data-a=listen]').onclick = e => toggleSpeak(m.text, e.currentTarget);
    a.querySelector('[data-a=copy]').onclick = () => { navigator.clipboard?.writeText(m.text); toast('Copied'); };
    body.append(a);
  }
  return w;
}

function renderMessages() {
  const box = $('#messages'); box.innerHTML = '';
  const c = activeChat();
  if (!c.messages.length) box.append(welcome());
  c.messages.forEach(m => box.append(msgEl(m)));
  box.scrollTop = box.scrollHeight;
}
function refreshLast() { // re-render only the last message (cheap during streaming)
  const box = $('#messages'), c = activeChat(), m = c.messages[c.messages.length - 1];
  if (box.lastElementChild) box.lastElementChild.replaceWith(msgEl(m));
  box.scrollTop = box.scrollHeight;
}
function renderAll() { renderSidebar(); $('#chatTitle').textContent = activeChat().title; renderMessages(); }

/* ---------- text-to-speech for LLM responses ---------- */
let speakingBtn = null;
function toggleSpeak(text, btn) {
  if (!('speechSynthesis' in window)) return toast('browser speech not supported');
  const wasMe = speakingBtn === btn;
  speechSynthesis.cancel();
  if (speakingBtn) { speakingBtn.classList.remove('on'); speakingBtn.textContent = '🔊 Listen'; speakingBtn = null; }
  if (wasMe) return;
  const clean = text.replace(/```[\s\S]*?```/g, ' code block ').replace(/\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$/g, ' formula ').replace(/[*#`_>\[\]()]/g, '');
  const u = new SpeechSynthesisUtterance(clean);
  u.lang = /[\u0980-\u09FF]/.test(clean) ? 'bn-BD' : 'en-US';
  u.onend = u.onerror = () => { btn.classList.remove('on'); btn.textContent = '🔊 Listen'; if (speakingBtn === btn) speakingBtn = null; };
  btn.classList.add('on'); btn.textContent = '⏹ Stop'; speakingBtn = btn;
  speechSynthesis.speak(u);
}

/* ---------- attachments ---------- */
let pending = []; // {file, kind, url}
function addFiles(files, forceKind) {
  [...files].forEach(f => {
    const kind = forceKind || (f.type.startsWith('image/') ? 'image' : 'file');
    pending.push({ file: f, kind, url: kind === 'image' ? URL.createObjectURL(f) : null });
  });
  renderChips();
}
function renderChips() {
  const box = $('#chips'); box.innerHTML = '';
  pending.forEach((p, i) => {
    const c = el('div', 'chip', `${p.kind === 'image' ? `<img src="${p.url}">` : '📄'}<span>${esc(p.file.name)}</span><button title="Remove">✕</button>`);
    c.querySelector('button').onclick = () => { pending.splice(i, 1); renderChips(); };
    box.append(c);
  });
}

/* ---------- voice recording → transcription ---------- */
let rec = null, recChunks = [], recTimer = null;
async function toggleRecord() {
  const btn = $('#micBtn');
  if (rec && rec.state === 'recording') return rec.stop();
  if (!navigator.mediaDevices?.getUserMedia) return toast('Mic is not support(HTTPS/localhost need)');
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); } catch { return toast('Mic permission'); }
  recChunks = []; rec = new MediaRecorder(stream);
  rec.ondataavailable = e => e.data.size && recChunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach(t => t.stop()); clearInterval(recTimer);
    btn.classList.remove('rec'); btn.textContent = '🎙️';
    const blob = new Blob(recChunks, { type: rec.mimeType || 'audio/webm' });
    if (blob.size) await transcribe(blob);
  };
  rec.start();
  let s = 0; btn.classList.add('rec'); btn.textContent = '⏹ 0s';
  recTimer = setInterval(() => { btn.textContent = `⏹ ${++s}s`; }, 1000);
}
async function transcribe(blob) {
  toast('Transcribing…');
  try {
    const fd = new FormData(); fd.append('audio', blob, 'voice.webm');
    const res = await fetch(CONFIG.API_BASE + '/api/transcribe', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(res.status);
    insertText((await res.json()).text || '');
  } catch {
    if (!CONFIG.MOCK_FALLBACK) return toast('Transcription failed');
    insertText('(demo transcript — /api/transcribe connect)');
  }
}
function insertText(t) { const i = $('#input'); i.value = (i.value ? i.value + ' ' : '') + t; autosize(); i.focus(); }
function autosize() { const i = $('#input'); i.style.height = 'auto'; i.style.height = Math.min(i.scrollHeight, 160) + 'px'; }

/* ---------- workflow graph (SVG animation) ---------- */
const NODES = {
  prepare_query:         { x: 260, y: 36,  t: 'router', label: 'Prepare Query (image + history)' },
  check_semantic_cache:  { x: 260, y: 112, t: 'router', label: 'Check Semantic Cache' },
  route_query:           { x: 260, y: 190, t: 'router', label: 'Route Query (RAG / Web / Direct)' },
  retrieve:              { x: 100, y: 275, t: 'rag',    label: 'Retrieve from Vector DB' },
  generate_direct:       { x: 410, y: 275, t: 'direct', label: 'Direct Answer' },
  web_search:            { x: 410, y: 355, t: 'web',    label: 'Web Search (text + images)' },
  no_answer_found:       { x: 410, y: 435, t: 'fail',   label: 'No Answer Found' },
  check_retrieval_score: { x: 100, y: 355, t: 'rag',    label: 'Filter by Similarity' },
  query_optimizer:       { x: 260, y: 355, t: 'rag',    label: 'Optimize Query' },
  generate_from_context: { x: 100, y: 435, t: 'rag',    label: 'Generate from Context' },
  check_grounded:        { x: 100, y: 515, t: 'rag',    label: 'Groundedness Check' },
  revise_answer:         { x: 260, y: 515, t: 'rag',    label: 'Revise with Quotes' },
  check_useful:          { x: 100, y: 595, t: 'rag',    label: 'Usefulness Check' },
  evaluate_answer:       { x: 260, y: 595, t: 'eval',   label: 'Evaluate Answer' },
  rewrite_question:      { x: 100, y: 675, t: 'rag',    label: 'Rewrite Query' },
  save_cache:            { x: 260, y: 675, t: 'eval',   label: 'Save to Semantic Cache' },
  finalize:              { x: 260, y: 755, t: 'eval',   label: 'Finalize Response' },
};
// [from, to, optional custom SVG path]
const EDGES = [
  ['prepare_query', 'check_semantic_cache'],
  ['check_semantic_cache', 'finalize', 'M194 112 H20 V755 H194'],            // cache hit shortcut
  ['check_semantic_cache', 'route_query'],
  ['route_query', 'retrieve'], ['route_query', 'web_search'], ['route_query', 'generate_direct'],
  ['retrieve', 'check_retrieval_score'],
  ['check_retrieval_score', 'generate_from_context'],
  ['check_retrieval_score', 'query_optimizer'],
  ['query_optimizer', 'retrieve'],
  ['check_retrieval_score', 'web_search', 'M100 372 V392 H410 V372'],         // optional web fallback
  ['check_retrieval_score', 'no_answer_found', 'M100 372 V404 H318 V435 H344'],
  ['generate_from_context', 'check_grounded'],
  ['check_grounded', 'check_useful'],
  ['check_grounded', 'revise_answer'], ['revise_answer', 'check_grounded'],
  ['check_grounded', 'no_answer_found'],
  ['check_useful', 'evaluate_answer'],
  ['check_useful', 'rewrite_question'],
  ['rewrite_question', 'retrieve', 'M34 675 H6 V275 H34'],                    // loop back
  ['check_useful', 'no_answer_found', 'M100 578 V556 H410 V452'],
  ['evaluate_answer', 'save_cache'], ['save_cache', 'finalize'],
  ['generate_direct', 'finalize', 'M476 275 H492 V755 H326'],
  ['web_search', 'finalize', 'M476 355 H492 V755 H326'],
  ['no_answer_found', 'finalize', 'M476 435 H492 V755 H326'],
];
const NW = 132, NH = 34;
const hasEdge = (a, b) => EDGES.some(e => e[0] === a && e[1] === b);

function edgePath(a, b, custom) {
  if (custom) return custom;
  const A = NODES[a], B = NODES[b], hw = NW / 2, hh = NH / 2;
  if (Math.abs(B.y - A.y) < 60) { // same row: side to side (offset when a reverse edge exists)
    const dir = B.x > A.x ? 1 : -1, sx = A.x + dir * hw, ex = B.x - dir * hw, mx = (sx + ex) / 2;
    const off = hasEdge(b, a) ? (a < b ? -6 : 6) : 0;
    return `M${sx} ${A.y + off} C${mx} ${A.y + off}, ${mx} ${B.y + off}, ${ex} ${B.y + off}`;
  }
  const up = B.y < A.y, sy = A.y + (up ? -hh : hh), ey = B.y + (up ? hh : -hh), my = (sy + ey) / 2;
  return `M${A.x} ${sy} C${A.x} ${my}, ${B.x} ${my}, ${B.x} ${ey}`;
}

const graph = {
  svg: null, visited: new Set(), active: new Set(), t0: 0, starts: {},
  build() {
    this.svg = $('#graph');
    let h = `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#4a5670"/></marker></defs>`;
    EDGES.forEach(([a, b, s]) => { h += `<path class="edge" data-e="${a}>${b}" d="${edgePath(a, b, s)}"/>`; });
    Object.entries(NODES).forEach(([id, n]) => {
      h += `<g class="node ${n.t}" data-id="${id}" transform="translate(${n.x},${n.y})"><rect x="${-NW / 2}" y="${-NH / 2}" width="${NW}" height="${NH}" rx="9"/><title>${esc(n.label || id)}</title><text>${id}</text></g>`;
    });
    this.svg.innerHTML = h;
  },
  reset() {
    this.visited.clear(); this.active.clear(); this.starts = {}; this.last = null; this.t0 = performance.now();
    this.svg.querySelectorAll('.node').forEach(n => n.classList.remove('active', 'done'));
    this.svg.querySelectorAll('.edge').forEach(e => e.classList.remove('flow', 'done'));
    $('#graphLog').innerHTML = '';
    this.status('Idle');
  },
  status(t, cls = '') { const s = $('#graphStatus'); s.textContent = t; s.className = 'status ' + cls; },
  node(id) { return this.svg.querySelector(`.node[data-id="${id}"]`); },
  start(id) {
    if (!NODES[id]) {
      console.warn('unknown node', id); this.active.add(id); this.starts[id] = performance.now();
      this.status('Running: ' + id, 'running');
      $('#graphLog').insertAdjacentHTML('beforeend', `<div>▶ ${esc(id)} <i>(diagram-এ নেই)</i></div>`); return;
    }
    const direct = this.last && hasEdge(this.last, id);
    EDGES.forEach(([a, b]) => { // retrieve/check_grounded loop
      if (b !== id) return;
      if (direct ? a === this.last : this.visited.has(a)) {
        const e = this.svg.querySelector(`[data-e="${a}>${b}"]`);
        e?.classList.remove('done'); e?.classList.add('flow');
      }
    });
    this.last = id;
    this.node(id).classList.add('active'); this.node(id).classList.remove('done');
    this.active.add(id); this.visited.add(id); this.starts[id] = performance.now();
    this.status('Running: ' + id, 'running');
    this.node(id).scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
  },
  end(id) {
    if (!NODES[id]) {
      this.active.delete(id);
      const t = ((performance.now() - (this.starts[id] || performance.now())) / 1000).toFixed(2);
      $('#graphLog').insertAdjacentHTML('beforeend', `<div>✓ ${esc(id)} <b>${t}s</b></div>`); return;
    }
    const n = this.node(id); n.classList.remove('active'); n.classList.add('done'); this.active.delete(id);
    this.svg.querySelectorAll(`[data-e$=">${id}"]`).forEach(e => { if (e.classList.contains('flow')) { e.classList.remove('flow'); e.classList.add('done'); } });
    const dt = ((performance.now() - (this.starts[id] || performance.now())) / 1000).toFixed(2);
    const log = $('#graphLog'); log.insertAdjacentHTML('beforeend', `<div>✓ ${id} <b>${dt}s</b></div>`); log.scrollTop = log.scrollHeight;
  },
  event(ev) { // supports status start/end, or bare events (auto-closes previous node)
    if (ev.status === 'end') return this.end(ev.name);
    if (!ev.status) [...this.active].forEach(a => this.end(a));
    this.start(ev.name);
  },
  finish() {
    [...this.active].forEach(a => this.end(a));
    this.status(`Finished in ${((performance.now() - this.t0) / 1000).toFixed(1)}s`, 'finished');
  },
};

/* ---------- chat request (stream) ---------- */
async function runChat(fd, query, onEvent) {
  try {
    const res = await fetch(CONFIG.API_BASE + '/api/chat', { method: 'POST', body: fd });
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim().replace(/^data:\s*/, ''); buf = buf.slice(i + 1);
        if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch { console.warn('bad line', line); continue; }
        onEvent(ev);
      }
    }
  } catch (err) {
    if (!CONFIG.MOCK_FALLBACK) throw err;
    toast('Backend failed → demo mode');
    await mockRun(query, onEvent);
  }
}

/* ---------- send message ---------- */
let sending = false;
async function sendMessage() {
  if (sending) return;
  const input = $('#input'), text = input.value.trim();
  if (!text && !pending.length) return;
  const chat = activeChat();
  const files = pending; pending = []; renderChips();
  input.value = ''; autosize();

  chat.messages.push({ role: 'user', text, attachments: files.map(p => ({ name: p.file.name, kind: p.kind, url: p.url })) });
  if (chat.title === 'New chat') chat.title = (text || files[0].file.name).slice(0, 42);
  const reply = { role: 'assistant', text: '', media: [], sources: [], pending: true };
  chat.messages.push(reply); chat.updatedAt = Date.now();
  renderAll();

  const fd = new FormData();
  fd.append('query', text); fd.append('chat_id', chat.id); fd.append('project_id', chat.projectId || chat.id);
  files.forEach(p => fd.append(p.kind === 'image' ? 'images' : 'files', p.file));

  sending = true; $('#sendBtn').disabled = true;
  $('#app').classList.add('graph-open'); graph.reset();
  let acc = '';
  try {
    await runChat(fd, text, ev => {
      if (ev.type === 'node') graph.event(ev);
      else if (ev.type === 'token') { acc += ev.text; reply.text = acc; refreshLast(); }
      else if (ev.type === 'final') {
        reply.text = ev.answer ?? acc; reply.media = ev.media || []; reply.sources = ev.sources || []; reply.pending = false; refreshLast();
      }
    });
  } catch (err) {
    reply.text = '⚠️ Request failed: ' + err.message;
  } finally {
    reply.pending = false; graph.finish();
    sending = false; $('#sendBtn').disabled = false;
    save(); refreshLast(); renderSidebar();
  }
}

/* ---------- image drawer (sidebar) ---------- */
let currentMedia = null;
function openDrawer(m) {
  currentMedia = m;
  $('#drawerTitle').textContent = m.type === 'chart' ? 'Chart' : 'Image';
  $('#drawerImg').src = m.url; $('#drawerCap').textContent = m.caption || '';
  const links = $('#drawerLinks'); links.innerHTML = '';
  links.insertAdjacentHTML('beforeend', `<a href="${esc(m.url)}" target="_blank" rel="noopener">Open original</a><a href="${esc(m.url)}" download>Download</a>`);
  if (m.source) links.insertAdjacentHTML('beforeend', `<a href="${esc(m.source)}" target="_blank" rel="noopener">Source page</a>`);
  $('#drawerThread').innerHTML = '';
  $('#drawer').classList.add('open'); $('#drawerInput').focus();
}
function closeDrawer() { $('#drawer').classList.remove('open'); }
function threadMsg(role, html) { const d = el('div', 'tmsg ' + role, html); $('#drawerThread').append(d); d.scrollIntoView({ block: 'end' }); return d; }

async function askAboutImage() {
  const inp = $('#drawerInput'), q = inp.value.trim();
  if (!q || !currentMedia) return;
  inp.value = ''; threadMsg('user', esc(q));
  const out = threadMsg('assistant', '<div class="typing"><i></i><i></i><i></i></div>');
  const fd = new FormData();
  fd.append('query', q); fd.append('chat_id', activeChat().id); fd.append('project_id', activeChat().projectId || activeChat().id); fd.append('context_image', currentMedia.url);
  let acc = '';
  try {
    await runChat(fd, q, ev => {
      if (ev.type === 'node') graph.event(ev);
      else if (ev.type === 'token') { acc += ev.text; out.innerHTML = md(acc); }
      else if (ev.type === 'final') out.innerHTML = md(ev.answer ?? acc);
    });
  } catch (e) { out.textContent = '⚠️ ' + e.message; }
}

/* ---------- mock backend (for UI testing without FastAPI) ---------- */
function svgChart(title, vals, color) {
  const bars = vals.map((v, i) => `<rect x="${30 + i * 40}" y="${130 - v}" width="26" height="${v}" rx="4" fill="${color}"/>`).join('');
  const s = `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="170"><rect width="320" height="170" fill="#1b2331"/><text x="14" y="20" fill="#e6edf3" font-family="sans-serif" font-size="13">${title}</text>${bars}<line x1="20" y1="130" x2="300" y2="130" stroke="#3a465c"/></svg>`;
  return 'data:image/svg+xml;utf8,' + encodeURIComponent(s);
}
async function mockRun(q, emit) {
  const path = ['prepare_query', 'check_semantic_cache', 'route_query', 'retrieve', 'check_retrieval_score', 'generate_from_context', 'check_grounded', 'check_useful', 'evaluate_answer', 'save_cache', 'finalize'];
  for (const n of path) { emit({ type: 'node', name: n, status: 'start' }); await sleep(650); emit({ type: 'node', name: n, status: 'end' }); }
  const answer = `**demo answer** : "${q || 'image/file'}".\n\n- Backend connect -> RAG answer\n- image click -> sidebar open\n- 🔊 Listen click -> Voice (answer)`;
  for (const w of answer.split(' ')) { emit({ type: 'token', text: w + ' ' }); await sleep(22); }
  emit({
    type: 'final', answer,
    media: [
      { type: 'chart', url: svgChart('Revenue (demo)', [40, 70, 55, 95, 80], '#6d8dff'), caption: 'Revenue chart' },
      { type: 'image', url: svgChart('Accuracy (demo)', [60, 85, 75, 100, 90], '#22d3a6'), caption: 'Accuracy chart', source: 'https://example.com' },
      { type: 'table', columns: ['Model', 'Score'], rows: [['A', '0.82'], ['B', '0.91']] },
    ],
    sources: [{ title: 'example.com', url: 'https://example.com' }],
  });
}

/* ---------- mobile helpers ---------- */
function closeMobile() { $('#sidebar').classList.remove('open'); $('#backdrop').classList.remove('show'); }

/* ---------- wire up ---------- */
function init() {
  graph.build();
  if (!state.chats.length) newChat(); else { if (!activeChat()) state.activeId = state.chats[0].id; renderAll(); }

  $('#newChatBtn').onclick = () => newChat();
  $('#newProjectBtn').onclick = newProject;
  $('#searchInput').oninput = renderSidebar;
  $('#sendBtn').onclick = sendMessage;
  $('#attachBtn').onclick = () => $('#fileInput').click();
  $('#imageBtn').onclick = () => $('#imageInput').click();
  $('#fileInput').onchange = e => { addFiles(e.target.files); e.target.value = ''; };
  $('#imageInput').onchange = e => { addFiles(e.target.files, 'image'); e.target.value = ''; };
  $('#micBtn').onclick = toggleRecord;
  $('#toggleGraphBtn').onclick = () => $('#app').classList.toggle('graph-open');
  $('#drawerClose').onclick = closeDrawer;
  $('#drawerSend').onclick = askAboutImage;
  $('#drawerInput').onkeydown = e => { if (e.key === 'Enter') askAboutImage(); };
  $('#menuBtn').onclick = () => { $('#sidebar').classList.add('open'); $('#backdrop').classList.add('show'); };
  $('#backdrop').onclick = closeMobile;
  $('#chatTitle').ondblclick = () => renameChat(activeChat());

  const input = $('#input');
  input.oninput = autosize;
  input.onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendMessage(); } };
  input.onpaste = e => { const imgs = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith('image/')); if (imgs.length) { e.preventDefault(); addFiles(imgs, 'image'); } };

  const comp = $('#composer');
  ['dragenter', 'dragover'].forEach(t => comp.addEventListener(t, e => { e.preventDefault(); comp.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(t => comp.addEventListener(t, e => { e.preventDefault(); comp.classList.remove('drag'); }));
  comp.addEventListener('drop', e => addFiles(e.dataTransfer.files));

  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
}
init();
