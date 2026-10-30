// CoalOS - the screen of Coal's pretend computer. The disk is the server (coalos/store.py).
// Limits are part of the design: 512 MB of memory, one slow core, a 2 GB disk.
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---- the disk -----------------------------------------------------------------
const api = {
  async ls(path) { return (await fetch(`/api/ls?path=${encodeURIComponent(path)}`)).json(); },
  async text(path) { const r = await fetch(`/api/file?path=${encodeURIComponent(path)}`); if (!r.ok) throw new Error((await r.json()).error); return r.text(); },
  url: path => `/api/file?path=${encodeURIComponent(path)}`,
  async post(route, body) {
    const r = await fetch(`/api/${route}`, { method: "POST", body: JSON.stringify(body) });
    const j = await r.json(); if (!r.ok) throw new Error(j.error); return j;
  },
  async upload(dir, file) {
    const r = await fetch(`/api/upload?dir=${encodeURIComponent(dir)}&name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    const j = await r.json(); if (!r.ok) throw new Error(j.error); return j;
  },
  async stats() { return (await fetch("/api/stats")).json(); },
};
const join = (dir, name) => (dir === "/" ? "" : dir) + "/" + name;
const parent = p => p.replace(/\/[^/]+$/, "") || "/";
const base = p => p.split("/").pop();
const changed = () => document.dispatchEvent(new Event("disk"));     // every view refreshes itself

// ---- icons (drawn here, nothing borrowed) ---------------------------------------
const ICON = {
  folder: '<svg viewBox="0 0 32 32"><path d="M2 8h11l3 3h14v16H2z" fill="#e8b93a"/><path d="M2 12h28v15H2z" fill="#f8d775"/><path d="M2 12h28" stroke="#c9952a"/></svg>',
  file: '<svg viewBox="0 0 32 32"><path d="M7 2h13l6 6v22H7z" fill="#fff" stroke="#7a8aa6"/><path d="M20 2v6h6" fill="#dfe6f3" stroke="#7a8aa6"/></svg>',
  text: '<svg viewBox="0 0 32 32"><path d="M7 2h13l6 6v22H7z" fill="#fff" stroke="#7a8aa6"/><path d="M20 2v6h6" fill="#dfe6f3" stroke="#7a8aa6"/><path d="M10 13h12M10 17h12M10 21h9" stroke="#3a5da8"/></svg>',
  image: '<svg viewBox="0 0 32 32"><rect x="3" y="5" width="26" height="22" rx="2" fill="#fff" stroke="#7a8aa6"/><path d="M5 25l7-9 5 6 4-4 6 7z" fill="#5aa83a"/><circle cx="22" cy="11" r="3" fill="#f4c430"/></svg>',
  music: '<svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="#2f6fd6"/><path d="M13 22V9l10-2v12" stroke="#fff" stroke-width="2" fill="none"/><circle cx="11" cy="22" r="3" fill="#fff"/><circle cx="21" cy="19" r="3" fill="#fff"/></svg>',
  video: '<svg viewBox="0 0 32 32"><rect x="3" y="6" width="26" height="20" rx="3" fill="#333"/><path d="M13 11l8 5-8 5z" fill="#fff"/></svg>',
  code: '<svg viewBox="0 0 32 32"><rect x="3" y="4" width="26" height="24" rx="3" fill="#1e1e1e"/><path d="M12 11l-5 5 5 5M20 11l5 5-5 5" stroke="#3ba7f5" stroke-width="2.4" fill="none"/></svg>',
  terminal: '<svg viewBox="0 0 32 32"><rect x="3" y="5" width="26" height="22" rx="2" fill="#000" stroke="#888"/><path d="M8 12l5 4-5 4M15 21h8" stroke="#c0c0c0" stroke-width="2" fill="none"/></svg>',
  computer: '<svg viewBox="0 0 32 32"><rect x="4" y="4" width="24" height="17" rx="2" fill="#dfe6f3" stroke="#555"/><rect x="7" y="7" width="18" height="11" fill="#3a6fd6"/><path d="M11 27h10l-2-6h-6z" fill="#aaa"/></svg>',
  info: '<svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="#2f6fd6"/><path d="M16 14v9" stroke="#fff" stroke-width="3"/><circle cx="16" cy="9.5" r="2" fill="#fff"/></svg>',
  notepad: '<svg viewBox="0 0 32 32"><rect x="6" y="4" width="20" height="25" fill="#fff" stroke="#7a8aa6"/><rect x="6" y="4" width="20" height="5" fill="#6fa8f0"/><path d="M9 14h14M9 18h14M9 22h10" stroke="#999"/></svg>',
};
const kindOf = name => {
  const e = (name.split(".").pop() || "").toLowerCase();
  if (/^(jpe?g|png|gif|webp|bmp|svg)$/.test(e)) return "image";
  if (/^(mp3|m4a|aac|ogg|opus|wav|flac)$/.test(e)) return "music";
  if (/^(mp4|webm|mkv|mov)$/.test(e)) return "video";
  if (/^(py|js|ts|html|css|json|c|cpp|java|sh)$/.test(e)) return "code";
  if (/^(txt|md|log|csv|ini)$/.test(e) || !name.includes(".")) return "text";
  return "file";
};
const icon = k => `<span class="ico">${ICON[k] || ICON.file}</span>`;
document.querySelectorAll("[data-ico]").forEach(s => s.innerHTML = ICON[s.dataset.ico]);

// ---- limits: memory and processor -----------------------------------------------
const RAM_TOTAL = 512;                       // MB, pretend
const COST = { explorer: 24, notepad: 8, terminal: 12, code: 96, viewer: 40, player: 48, about: 4 };
const RUN_COST = 128, CPU_SECONDS = 5;       // running code: more memory, and a time limit
let ramUsed = 64, cpuBusy = 0;                // 64 MB = the system itself
function balloon(title, text) {
  const b = document.createElement("div"); b.className = "balloon"; b.innerHTML = `<b>${esc(title)}</b>${esc(text)}`;
  document.body.append(b); setTimeout(() => b.remove(), 5000);
}
function take(mb, what) {
  if (ramUsed + mb > RAM_TOTAL) { balloon("Not enough memory", `${what} needs ${mb} MB, only ${RAM_TOTAL - ramUsed} MB free. Close something first.`); return false; }
  ramUsed += mb; meters(); return true;
}
function give(mb) { ramUsed = Math.max(64, ramUsed - mb); meters(); }
function meters() {
  const r = ramUsed / RAM_TOTAL, c = Math.max(cpuBusy, 0.04 + Math.random() * 0.04);
  $("#ram i").style.width = `${r * 100}%`; $("#ram").classList.toggle("hot", r > 0.85);
  $("#ram").title = `Memory: ${ramUsed} of ${RAM_TOTAL} MB in use (pretend)`;
  $("#cpu i").style.width = `${c * 100}%`; $("#cpu").classList.toggle("hot", c > 0.85);
}
setInterval(meters, 1000);

// ---- windows ----------------------------------------------------------------------
let z = 10, offset = 0;
const wins = new Set();
function makeWin({ title, ico, app, w = 560, h = 380 }) {
  if (!take(COST[app] || 8, title)) return null;
  const el = document.createElement("div");
  el.className = "win";
  offset = (offset + 28) % 200;
  Object.assign(el.style, { left: `${80 + offset}px`, top: `${40 + offset}px`, width: `${w}px`, height: `${h}px` });
  el.innerHTML = `<div class="title">${icon(ico)}<span class="t">${esc(title)}</span><button class="mn" title="Minimize">_</button><button class="mx" title="Maximize">□</button><button class="x" title="Close">✕</button></div>`;
  const body = document.createElement("div"); body.style.cssText = "flex:1;display:flex;flex-direction:column;min-height:0";
  el.append(body);
  $("#windows").append(el);
  const task = document.createElement("div"); task.className = "task"; task.innerHTML = `${icon(ico)}<span>${esc(title)}</span>`;
  $("#tasks").append(task);
  const win = { el, body, task, app, onclose: [], setTitle(t) { $(".t", el).textContent = t; $("span:last-child", task).textContent = t; } };
  const focus = () => { wins.forEach(o => { o.el.classList.add("inactive"); o.task.classList.remove("on"); }); el.classList.remove("inactive"); el.hidden = false; el.style.zIndex = ++z; task.classList.add("on"); };
  win.focus = focus;
  el.addEventListener("pointerdown", focus);
  task.onclick = () => (el.hidden || el.classList.contains("inactive")) ? focus() : (el.hidden = true, task.classList.remove("on"));
  $(".mn", el).onclick = e => { e.stopPropagation(); el.hidden = true; task.classList.remove("on"); };
  $(".mx", el).onclick = () => el.classList.toggle("max");
  $(".x", el).onclick = () => { win.onclose.forEach(f => f()); el.remove(); task.remove(); wins.delete(win); give(COST[app] || 8); };
  const bar = $(".title", el);
  bar.ondblclick = () => el.classList.toggle("max");
  bar.addEventListener("pointerdown", e => {
    if (e.target.closest("button") || el.classList.contains("max")) return;
    const sx = e.clientX - el.offsetLeft, sy = e.clientY - el.offsetTop;
    const move = m => { el.style.left = `${m.clientX - sx}px`; el.style.top = `${Math.max(0, m.clientY - sy)}px`; };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  });
  wins.add(win); focus();
  return win;
}

// ---- dropping real files in -----------------------------------------------------------
async function dropInto(dir, dt) {
  const items = [...(dt.items || [])].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  let n = 0;
  const walk = async (entry, into) => {
    if (entry.isFile) { const f = await new Promise((ok, no) => entry.file(ok, no)); await api.upload(into, f); n++; }
    else { const d = join(into, entry.name); await api.post("mkdir", { path: d });
      const reader = entry.createReader(); let batch;
      while ((batch = await new Promise(ok => reader.readEntries(ok))).length) for (const e of batch) await walk(e, d); }
  };
  try {
    if (items.length) for (const e of items) await walk(e, dir);
    else for (const f of dt.files) { await api.upload(dir, f); n++; }
    balloon("Copied in", `${n} file${n === 1 ? "" : "s"} now live in ${dir}.`);
  } catch (e) { balloon("Could not copy", e.message); }
  changed();
}
function dropZone(el, dirOf, cls = "over") {
  el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add(cls); });
  el.addEventListener("dragleave", e => { if (!el.contains(e.relatedTarget)) el.classList.remove(cls); });
  el.addEventListener("drop", async e => {
    e.preventDefault(); e.stopPropagation(); el.classList.remove(cls);
    const inner = e.dataTransfer.getData("text/coalos");
    if (inner) { try { await api.post("move", { src: inner, dst: dirOf(), copy: e.ctrlKey }); } catch (x) { balloon("Could not move", x.message); } changed(); }
    else if (e.dataTransfer.files.length) dropInto(dirOf(), e.dataTransfer);
  });
}
document.addEventListener("paste", e => {
  if (e.target.closest("textarea, input")) return;
  if (e.clipboardData.files.length) dropInto(window.pasteDir || "/home/Desktop", e.clipboardData);
});

// ---- right-click menu ----------------------------------------------------------------
function menu(e, items) {
  e.preventDefault();
  const m = $("#menu"); m.innerHTML = "";
  items.forEach(it => { if (it === "-") return m.append(document.createElement("hr"));
    const d = document.createElement("div"); d.textContent = it[0]; d.onclick = () => { m.hidden = true; it[1](); }; m.append(d); });
  Object.assign(m.style, { left: `${Math.min(e.clientX, innerWidth - 170)}px`, top: `${Math.min(e.clientY, innerHeight - 40 - items.length * 24)}px` });
  m.hidden = false;
}
addEventListener("pointerdown", e => { if (!e.target.closest("#menu")) $("#menu").hidden = true; if (!e.target.closest("#start-menu, #start")) $("#start-menu").hidden = true; });
function itemMenu(e, entry) {
  menu(e, [
    ["Open", () => open(entry.path, entry.kind)],
    ["Rename", async () => { const n = prompt("New name:", entry.name); if (n && n !== entry.name) { try { await api.post("move", { src: entry.path, dst: join(parent(entry.path), n) }); } catch (x) { balloon("Could not rename", x.message); } changed(); } }],
    ...(entry.kind === "file" ? [["Download to the real computer", () => { const a = document.createElement("a"); a.href = api.url(entry.path) + "&download=1"; a.click(); }]] : []),
    "-",
    ["Delete", async () => { try { await api.post("remove", { path: entry.path }); } catch (x) { balloon("Could not delete", x.message); } changed(); }],
  ]);
}
function newMenu(e, dir) {
  menu(e, [
    ["New folder", async () => { const n = prompt("Folder name:", "New Folder"); if (n) { await api.post("mkdir", { path: join(dir, n) }).catch(x => balloon("Could not", x.message)); changed(); } }],
    ["New text document", async () => { const n = prompt("File name:", "New Text Document.txt"); if (n) { await api.post("write", { path: join(dir, n), text: "" }).catch(x => balloon("Could not", x.message)); changed(); } }],
  ]);
}

// ---- opening things --------------------------------------------------------------------
function open(path, kind = "file") {
  if (kind === "dir") return explorer(path);
  const k = kindOf(base(path));
  if (k === "image" || k === "video") return viewer(path, k);
  if (k === "music") return player(path);
  if (k === "code") return code(path);
  return notepad(path);
}

function explorer(path = "/home") {
  const w = makeWin({ title: base(path) || "My Computer", ico: "folder", app: "explorer", w: 640, h: 420 }); if (!w) return;
  let dir = path, sel = null;
  w.body.innerHTML = `<div class="menubar"><span>File</span><span>Edit</span><span>View</span><span>Help</span></div>
    <div class="addr"><button class="up">⬆ Up</button>Address <input></div>
    <div class="body"><div class="side">
      <div class="pane"><h4>File and Folder Tasks</h4><a class="nf">Make a new folder</a><a class="nt">Make a new text file</a></div>
      <div class="pane"><h4>Other Places</h4><a data-go="/home/Desktop">Desktop</a><a data-go="/home/Documents">My Documents</a><a data-go="/home/Pictures">My Pictures</a><a data-go="/home/Music">My Music</a><a data-go="/home/Code">Code</a></div>
      <div class="pane"><h4>Details</h4><p class="det">Drag files here from your real computer to copy them in.</p></div>
    </div><div class="grid"></div></div><div class="status"><span class="cnt"></span><span class="sz"></span></div>`;
  const grid = $(".grid", w.body), addr = $(".addr input", w.body);
  const go = d => { dir = d; window.pasteDir = d; draw(); };
  async function draw() {
    addr.value = dir; w.setTitle(base(dir) || "/");
    let list; try { list = await api.ls(dir); } catch { list = []; }
    grid.innerHTML = "";
    list.forEach(entry => {
      const k = entry.kind === "dir" ? "folder" : kindOf(entry.name);
      const it = document.createElement("div"); it.className = "item"; it.draggable = true;
      it.innerHTML = `${icon(k)}<span>${esc(entry.name)}</span>`;
      it.onclick = () => { grid.querySelectorAll(".sel").forEach(x => x.classList.remove("sel")); it.classList.add("sel"); sel = entry;
        $(".det", w.body).innerHTML = entry.kind === "dir" ? `<b>${esc(entry.name)}</b><br>Folder` :
          `<b>${esc(entry.name)}</b><br>${fmt(entry.size)}<br>Takes ${fmt(entry.stored)} on the laptop`; };
      it.ondblclick = () => entry.kind === "dir" ? go(entry.path) : open(entry.path);
      it.oncontextmenu = e => itemMenu(e, entry);
      it.ondragstart = e => e.dataTransfer.setData("text/coalos", entry.path);
      if (entry.kind === "dir") dropZone(it, () => entry.path, "target");
      grid.append(it);
    });
    $(".cnt", w.body).textContent = `${list.length} object${list.length === 1 ? "" : "s"}`;
  }
  grid.oncontextmenu = e => { if (e.target === grid) newMenu(e, dir); };
  dropZone(grid, () => dir);
  $(".up", w.body).onclick = () => go(parent(dir));
  addr.onkeydown = e => { if (e.key === "Enter") go(addr.value); };
  w.body.querySelectorAll("[data-go]").forEach(a => a.onclick = () => go(a.dataset.go));
  $(".nf", w.body).onclick = e => newMenu(e, dir);
  $(".nt", w.body).onclick = e => newMenu(e, dir);
  w.el.addEventListener("keydown", async e => { if (e.key === "Delete" && sel) { await api.post("remove", { path: sel.path }).catch(x => balloon("Could not delete", x.message)); changed(); } });
  w.el.tabIndex = 0;
  const redraw = () => draw(); document.addEventListener("disk", redraw); w.onclose.push(() => document.removeEventListener("disk", redraw));
  go(dir);
}
const fmt = n => n < 1024 ? `${n} bytes` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;

function notepad(path) {
  const w = makeWin({ title: `${path ? base(path) : "Untitled"} - Notepad`, ico: "notepad", app: "notepad", w: 520, h: 380 }); if (!w) return;
  w.body.innerHTML = `<div class="menubar"><span class="sv">File: Save (Ctrl+S)</span></div><textarea class="pad" spellcheck="false"></textarea>`;
  const ta = $("textarea", w.body);
  if (path) api.text(path).then(t => ta.value = t).catch(e => ta.value = e.message);
  const save = async () => { if (!path) { const n = prompt("Save as:", "/home/Documents/Untitled.txt"); if (!n) return; path = n; }
    try { await api.post("write", { path, text: ta.value }); w.setTitle(`${base(path)} - Notepad`); changed(); } catch (e) { balloon("Could not save", e.message); } };
  $(".sv", w.body).onclick = save;
  ta.onkeydown = e => { if (e.ctrlKey && e.key === "s") { e.preventDefault(); save(); } };
}

function viewer(path, k) {
  const w = makeWin({ title: base(path), ico: k, app: "viewer", w: 600, h: 440 }); if (!w) return;
  w.body.innerHTML = `<div class="viewer">${k === "image" ? `<img src="${api.url(path)}">` : `<video controls src="${api.url(path)}"></video>`}</div>`;
}

async function player(path) {
  const w = makeWin({ title: "Media Player", ico: "music", app: "player", w: 420, h: 360 }); if (!w) return;
  const songs = (await api.ls(parent(path))).filter(e => e.kind === "file" && kindOf(e.name) === "music");
  w.body.innerHTML = `<div class="player"><b class="now"></b><audio controls autoplay></audio><ol></ol></div>`;
  const au = $("audio", w.body), ol = $("ol", w.body);
  const play = p => { au.src = api.url(p); $(".now", w.body).textContent = base(p); ol.querySelectorAll("li").forEach(li => li.classList.toggle("on", li.dataset.p === p)); };
  songs.forEach(s => { const li = document.createElement("li"); li.textContent = s.name; li.dataset.p = s.path; li.onclick = () => play(s.path); ol.append(li); });
  play(path);
  w.onclose.push(() => au.pause());
}

// ---- running code: inside the browser's sandbox, with a time limit -------------------------
const JS_WORKER = `self.onmessage = e => { const out = (...a) => postMessage({ out: a.map(x => typeof x === "object" ? JSON.stringify(x) : String(x)).join(" ") });
  console.log = console.info = console.warn = console.error = out;
  try { const r = (0, eval)(e.data.code); if (r !== undefined) out(r); } catch (err) { out(String(err)); } postMessage({ done: 1 }); };`;
const PY_WORKER = `importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");
  let py; const ready = loadPyodide().then(p => { py = p; py.setStdout({ batched: s => postMessage({ out: s }) }); py.setStderr({ batched: s => postMessage({ out: s }) }); postMessage({ ready: 1 }); });
  self.onmessage = async e => { await ready; try { const r = py.runPython(e.data.code); if (r !== undefined && r !== null) postMessage({ out: String(r) }); } catch (err) { postMessage({ out: String(err) }); } postMessage({ done: 1 }); };`;
let pyWorker = null, pyReady = false;   // the time limit only counts once Python has started
function runCode(lang, src, print) {
  return new Promise(done => {
    if (!take(RUN_COST, "Running this program")) return done();
    let w;
    if (lang === "python") { if (!pyWorker) { pyWorker = new Worker(URL.createObjectURL(new Blob([PY_WORKER]))); pyReady = false; print("(starting Python for the first time - a few seconds)"); } w = pyWorker; }
    else w = new Worker(URL.createObjectURL(new Blob([JS_WORKER])));
    let timer = null;
    const finish = msg => { clearTimeout(timer); cpuBusy = 0; give(RUN_COST); if (msg) print(msg); w.onmessage = null;
      if (lang !== "python") w.terminate(); done(); };
    const start = () => { cpuBusy = 1; timer = setTimeout(() => { w.terminate(); if (lang === "python") { pyWorker = null; pyReady = false; }
      finish(`stopped: took more than ${CPU_SECONDS} s on this slow processor`); }, CPU_SECONDS * 1000); };
    w.onmessage = e => { if (e.data.ready) { pyReady = true; return start(); } if (e.data.out !== undefined) print(e.data.out); if (e.data.done) finish(); };
    w.onerror = e => finish(String(e.message));
    if (lang !== "python" || pyReady) start();
    w.postMessage({ code: src });
  });
}

function code(path) {
  const w = makeWin({ title: "Code", ico: "code", app: "code", w: 760, h: 480 }); if (!w) return;
  w.body.innerHTML = `<div class="code" style="position:relative"><div class="tree"></div><div class="ed"><div class="gutter"></div><textarea spellcheck="false"></textarea></div><div class="out"></div>
    <div class="bar"><button class="save">Save</button><button class="run">▶ Run</button></div></div>`;
  const ta = $("textarea", w.body), gut = $(".gutter", w.body), out = $(".out", w.body), tree = $(".tree", w.body);
  let cur = path;
  const lines = () => { gut.innerHTML = ta.value.split("\n").map((_, i) => i + 1).join("<br>"); gut.scrollTop = ta.scrollTop; };
  ta.oninput = lines; ta.onscroll = () => gut.scrollTop = ta.scrollTop;
  ta.onkeydown = e => { if (e.key === "Tab") { e.preventDefault(); ta.setRangeText("    ", ta.selectionStart, ta.selectionEnd, "end"); lines(); }
    if (e.ctrlKey && e.key === "s") { e.preventDefault(); save(); } };
  const load = async p => { cur = p; ta.value = p ? await api.text(p).catch(() => "") : ""; lines(); w.setTitle(`${p ? base(p) : "untitled"} - Code`); drawTree(); };
  const save = async () => { if (!cur) { const n = prompt("Save as:", "/home/Code/main.py"); if (!n) return; cur = n; }
    try { await api.post("write", { path: cur, text: ta.value }); changed(); w.setTitle(`${base(cur)} - Code`); } catch (e) { balloon("Could not save", e.message); } };
  async function drawTree() {
    const dir = cur ? parent(cur) : "/home/Code";
    const list = (await api.ls(dir)).filter(e => e.kind === "file");
    tree.innerHTML = `<h5>${esc(dir.toUpperCase())}</h5>`;
    list.forEach(e => { const d = document.createElement("div"); d.textContent = e.name; d.classList.toggle("on", e.path === cur); d.onclick = () => load(e.path); tree.append(d); });
  }
  $(".save", w.body).onclick = save;
  $(".run", w.body).onclick = async () => {
    const lang = (cur || "").endsWith(".js") ? "javascript" : "python";
    out.textContent = `> ${lang} ${cur ? base(cur) : ""}\n`;
    await runCode(lang, ta.value, s => out.textContent += s + "\n");
  };
  load(path || null);
}

function terminal() {
  const w = makeWin({ title: "Command Prompt", ico: "terminal", app: "terminal", w: 600, h: 360 }); if (!w) return;
  w.body.innerHTML = `<div class="term"><div class="log">CoalOS [pretend computer]\nType "help" to see the commands.\n\n</div><div class="in"><span class="ps"></span><input spellcheck="false"></div></div>`;
  const log = $(".log", w.body), inp = $("input", w.body), ps = $(".ps", w.body), term = $(".term", w.body);
  let cwd = "/home", hist = [], hi = 0;
  const say = s => { log.textContent += s + "\n"; term.scrollTop = term.scrollHeight; };
  const at = p => !p ? cwd : p.startsWith("/") ? p : join(cwd, p);
  const prompt_ = () => ps.textContent = `${cwd}> `;
  term.onclick = () => inp.focus();
  const cmds = {
    help: () => say("ls  cd  pwd  cat  echo  mkdir  touch  rm  mv  cp  python  node  free  df  whoami  clear\n(echo text > file  writes a file)"),
    pwd: () => say(cwd),
    whoami: () => say("coal"),
    clear: () => log.textContent = "",
    ls: async a => { const l = await api.ls(at(a[0])); say(l.map(e => e.kind === "dir" ? e.name + "/" : e.name).join("  ") || "(empty)"); },
    cd: async a => { const d = at(a[0] || "/home").replace(/\/+$/, "") || "/";
      const norm = d.split("/").reduce((s, x) => x === ".." ? s.slice(0, -1) : x && x !== "." ? [...s, x] : s, []);
      const p = "/" + norm.join("/"); const l = await api.ls(parent(p)).catch(() => []);
      if (p === "/" || l.some(e => e.path === p && e.kind === "dir")) cwd = p; else say(`cd: no such folder: ${a[0]}`); },
    cat: async a => say(await api.text(at(a[0])).catch(e => `cat: ${e.message}`)),
    echo: async (a, raw) => { const m = raw.match(/^echo\s+(.*?)\s*>\s*(\S+)$/);
      if (m) { await api.post("write", { path: at(m[2]), text: m[1] + "\n" }).catch(e => say(e.message)); changed(); } else say(a.join(" ")); },
    mkdir: async a => { await api.post("mkdir", { path: at(a[0]) }).catch(e => say(e.message)); changed(); },
    touch: async a => { await api.post("write", { path: at(a[0]), text: "" }).catch(e => say(e.message)); changed(); },
    rm: async a => { await api.post("remove", { path: at(a[0]) }).catch(e => say(e.message)); changed(); },
    mv: async a => { await api.post("move", { src: at(a[0]), dst: at(a[1]) }).catch(e => say(e.message)); changed(); },
    cp: async a => { await api.post("move", { src: at(a[0]), dst: at(a[1]), copy: true }).catch(e => say(e.message)); changed(); },
    python: async a => { const src = await api.text(at(a[0])).catch(e => null); if (src === null) return say(`python: can't open ${a[0]}`); await runCode("python", src, say); },
    node: async a => { const src = await api.text(at(a[0])).catch(e => null); if (src === null) return say(`node: can't open ${a[0]}`); await runCode("javascript", src, say); },
    free: () => say(`memory: ${ramUsed} MB used of ${RAM_TOTAL} MB`),
    df: async () => { const s = await api.stats(); say(`disk: ${fmt(s.size)} used of ${fmt(s.disk)} (takes ${fmt(s.stored)} on the real laptop)`); },
  };
  inp.onkeydown = async e => {
    if (e.key === "ArrowUp") { inp.value = hist[--hi] ?? inp.value; hi = Math.max(0, hi); return; }
    if (e.key === "ArrowDown") { hi = Math.min(hist.length, hi + 1); inp.value = hist[hi] ?? ""; return; }
    if (e.key !== "Enter") return;
    const raw = inp.value.trim(); inp.value = "";
    if (raw) { hist.push(raw); hi = hist.length; }
    queue = queue.then(async () => {             // commands run in order, even when typed faster than they finish
      say(`${cwd}> ${raw}`); if (!raw) return;
      const [c, ...a] = raw.split(/\s+/);
      if (cmds[c]) { try { await cmds[c](a, raw); } catch (x) { say(String(x.message || x)); } }
      else say(`'${c}' is not a command here. Type help.`);
      prompt_();
    });
  };
  let queue = Promise.resolve();
  prompt_(); setTimeout(() => inp.focus(), 50);
}

function about() {
  const w = makeWin({ title: "About CoalOS", ico: "info", app: "about", w: 420, h: 300 }); if (!w) return;
  api.stats().then(s => w.body.innerHTML = `<div class="about"><b>CoalOS</b> - a pretend computer for Coal.<br><br>
    Memory: ${RAM_TOTAL} MB &nbsp;·&nbsp; Processor: one slow core (programs stop after ${CPU_SECONDS} s)<br>
    Disk: ${fmt(s.size)} of ${fmt(s.disk)} used, ${s.files} files<br>
    On the real laptop it all takes <b>${fmt(s.stored)}</b> (text is compressed, copies are stored once).<br><br>
    Nothing in here can reach the real computer. Code runs inside the browser's own sandbox.</div>`);
}

// ---- desktop, start menu, clock ----------------------------------------------------------
const APPS = [["My Computer", "computer", () => explorer("/home")], ["Command Prompt", "terminal", terminal], ["Code", "code", () => code()],
  ["Notepad", "notepad", () => notepad()], ["Media Player", "music", () => explorer("/home/Music")]];
async function drawDesktop() {
  const box = $("#icons"); box.innerHTML = "";
  const add = (name, k, run, entry) => {
    const d = document.createElement("div"); d.className = "dicon"; d.innerHTML = `${icon(k)}<span class="name">${esc(name)}</span>`;
    d.onclick = () => { box.querySelectorAll(".sel").forEach(x => x.classList.remove("sel")); d.classList.add("sel"); };
    d.ondblclick = run;
    if (entry) { d.draggable = true; d.ondragstart = e => e.dataTransfer.setData("text/coalos", entry.path); d.oncontextmenu = e => { e.stopPropagation(); itemMenu(e, entry); };
      if (entry.kind === "dir") dropZone(d, () => entry.path, "sel"); }
    box.append(d);
  };
  APPS.slice(0, 3).forEach(([n, k, f]) => add(n, k, f));
  add("My Documents", "folder", () => explorer("/home/Documents"));
  (await api.ls("/home/Desktop")).forEach(e => add(e.name, e.kind === "dir" ? "folder" : kindOf(e.name), () => open(e.path, e.kind), e));
}
const desk = $("#desktop");
dropZone(desk, () => "/home/Desktop", "dropping");
desk.oncontextmenu = e => { if (e.target === desk || e.target.id === "icons") newMenu(e, "/home/Desktop"); };
document.addEventListener("disk", () => { drawDesktop(); disk(); });
$("#sm-apps").innerHTML = APPS.map((a, i) => `<div class="sm-item" data-i="${i}">${icon(a[1])}${esc(a[0])}</div>`).join("");
$("#sm-apps").onclick = e => { const it = e.target.closest("[data-i]"); if (it) { $("#start-menu").hidden = true; APPS[it.dataset.i][2](); } };
document.querySelectorAll(".sm-right [data-open]").forEach(el => el.onclick = () => { $("#start-menu").hidden = true; explorer(el.dataset.open); });
$(".sm-right [data-app=about]").onclick = () => { $("#start-menu").hidden = true; about(); };
$("#start").onclick = () => $("#start-menu").hidden = !$("#start-menu").hidden;
async function disk() { const s = await api.stats(); $("#disk").textContent = `${fmt(s.size)} / ${fmt(s.disk)}`; $("#disk").title = `Pretend disk used. On the real laptop: ${fmt(s.stored)}`; }
const tick = () => $("#clock").textContent = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
setInterval(tick, 10000); tick(); drawDesktop(); disk(); meters();
