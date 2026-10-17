/* Qwen Image Editor: the browser side.
 *
 * One canvas in the middle shows either the image being marked or a result. Below it, the composer holds the
 * reference images, the prompt and the Generate button. History runs down the right.
 *
 * Marks are drawn as SVG over a reference while editing and flattened into a copy of the image only when
 * generating, so the original stays clean and the marks stay editable. The model sees the coloured lines;
 * the prompt refers to them by colour ("the red circle").
 */

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------------------------------------------------------------- icons

const ICON_PATHS = {
  plus: "M12 5v14M5 12h14",
  x: "M6 6l12 12M18 6L6 18",
  image: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M15.5 8.5h.01",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  pen: "M4 20l4-1L19 8l-3-3L5 16l-1 4zM14 7l3 3",
  undo: "M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  info: "M12 3a9 9 0 1 0 .01 0M12 11v5M12 8h.01",
  sliders: "M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4",
  edit: "M4 20h4L19 9l-4-4L4 16v4zM13 7l4 4",
  layers: "M12 4l8 4-8 4-8-4 8-4zM4 12l8 4 8-4M4 16l8 4 8-4",
  seed: "M5 5h14v14H5zM9 9h.01M15 15h.01M15 9h.01M9 15h.01",
  restore: "M4 12a8 8 0 1 0 2.5-5.8M4 4v4h4",
  check: "M5 12l5 5 9-10",
};
const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${ICON_PATHS[name]}"/></svg>`;

// ---------------------------------------------------------------- state

const MARK_COLORS = [
  { name: "red", css: "#ff3b47" }, { name: "green", css: "#22d861" }, { name: "blue", css: "#3b82f6" },
  { name: "magenta", css: "#f23ccd" }, { name: "yellow", css: "#ffd028" }, { name: "cyan", css: "#22d3ee" },
];
// Tested with Qwen-Image 2.1 in the Markup editor: without it, the model tends to draw the marks back in.
const NO_MARKS = "Do not render the circles in the output.";

const S = {
  refs: [],          // [{ id, w, h, marks: [{ color, points: [[x, y], ...], closed }] }], points in image pixels
  history: [],       // generation records from the server, newest first
  current: null,     // the record on the canvas
  mode: "empty",     // empty | ref | result
  refIndex: null,    // the reference being marked
  view: "result",    // result | compare | side | diff
  against: null,     // compare target key
  status: {},
  job: null,         // the generation in progress
  opts: { size: "auto", w: 1024, h: 1024, seed: "", kv: true, hideMarks: true },
};
try { Object.assign(S.opts, JSON.parse(localStorage.getItem("qie.options") || "{}")); } catch {}
const saveOpts = () => localStorage.setItem("qie.options", JSON.stringify(S.opts));

const refUrl = id => `/files/inputs/${id}.png`;
const resultUrl = rec => `/files/results/${rec.id}.png`;

// ---------------------------------------------------------------- server

const api = {
  async json(url, opts) {
    const r = await fetch(url, opts);
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
    return j;
  },
  status: () => api.json("/api/status"),
  history: () => api.json("/api/history"),
  upload: body => api.json("/api/upload", { method: "POST", body }),
  deleteHistory: body => api.json("/api/history/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  generate: body => api.json("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
};

async function addRef(blob) {
  const j = await api.upload(blob);
  const ref = { id: j.id, w: j.width, h: j.height, marks: [] };
  S.refs.push(ref);
  return ref;
}

// ---------------------------------------------------------------- status

async function refreshStatus() {
  try { S.status = await api.status(); } catch { S.status = { state: "offline" }; }
  const s = S.status, el = $("#status");
  el.className = "status " + (s.state === "ready" ? (s.busy ? "busy" : "ready") : s.state === "idle" ? "idle"
    : s.state === "error" || s.state === "offline" ? "error" : "");
  $("#statusText").textContent =
    s.state === "needs_setup" ? "Setup needed" :
    s.state === "idle" ? "Idle · model unloaded" :
    s.state === "setting_up" ? "Setting up" :
    s.state === "loading" ? (s.load_pct != null ? `Loading model · ${s.load_pct}%` : "Loading model…") :
    s.state === "error" ? (/NotSetUp/.test(s.error) ? "Not set up yet: run  uv run qie setup" : "Model failed to load") :
    s.state === "offline" ? "Server not reachable" :
    s.busy ? "Generating" : `Ready · ${s.gpu}`;
  el.title = s.state === "idle" ? "Unloaded after a while without use, to free memory. The next image loads it again." :
    s.state === "error" ? s.error : s.state === "ready" ? `${s.offload} memory mode · ${s.attn} attention · loaded in ${s.load_s} s`
    : s.state === "loading" && s.load_stage ? s.load_stage : "";
  updateGenerateButton();
  renderSetup();
}
let stopped = false;  // after Quit: stop polling a server that's gone
async function pollStatus() {
  if (stopped) return;
  await refreshStatus();
  setTimeout(pollStatus, ["loading", "setting_up"].includes(S.status.state) ? 500 : 3000);
}

// ---------------------------------------------------------------- first-run setup

// Shown instead of the editor until the models are in place: hardware check, model files, license, download.
let setupPlan = null, setupPlanLoading = false;
const gb = n => (n / 1e9 >= 10 ? (n / 1e9).toFixed(1) : (n / 1e9).toFixed(2)) + " GB";

async function loadSetupPlan(comfyui = []) {
  setupPlanLoading = true;
  try {
    setupPlan = comfyui.length
      ? await api.json("/api/setup/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ comfyui }) })
      : await api.json("/api/setup");
    setupPlan.comfyui = comfyui;
  } catch (e) { setupPlan = { error: e.message }; }
  setupPlanLoading = false;
  renderSetup(true);
}

function renderSetup(force = false) {
  const box = $("#setup"), state = S.status.state;
  const show = state === "needs_setup" || state === "setting_up";
  box.hidden = !show;
  if (!show) { box.dataset.view = ""; return; }
  if (state === "setting_up") return renderSetupProgress();
  if (!setupPlan && !setupPlanLoading) {
    box.innerHTML = `<div class="setup-card"><h1>Getting ready</h1><p class="muted">Checking this computer…</p></div>`;
    return loadSetupPlan();
  }
  if (!force && box.dataset.view === "plan") return;  // don't wipe the form while someone's using it
  if (!setupPlan) return;
  box.dataset.view = "plan";
  if (setupPlan.error) { box.innerHTML = `<div class="setup-card"><h1>Setup</h1><p class="error">${esc(setupPlan.error)}</p></div>`; return; }

  const p = setupPlan, hw = p.hardware, mac = p.backend === "mlx";
  const mark = { supported: "ok", experimental: "note", unsupported: "bad" }[hw.verdict];
  const facts = [hw.gpu && `${esc(hw.gpu)}${hw.vram_gib ? ` · ${hw.vram_gib} GB` : ""}`, hw.ram_gib && `${hw.ram_gib} GB memory`,
                 hw.free_disk_gib != null && `${hw.free_disk_gib} GB free disk`].filter(Boolean).join(" · ");
  const toGet = p.models.filter(m => !m.path);
  const converting = toGet.some(m => m.convert_from);
  const action = !toGet.length ? "Finish setup" : converting ? "Convert and set up"
    : mac ? `Download ${gb(p.download_bytes)} and convert` : `Download ${gb(p.download_bytes)} and set up`;
  const models = p.models.map(m => `<div class="model ${m.path ? "found" : ""}">
      <span class="mi">${m.path ? icon("check") : icon("download")}</span>
      <div><div>${esc(m.label)}</div><div class="muted small">${m.path ? `Found (${esc(m.source)}): ${esc(m.path)}`
        : m.convert_from ? `Converted for this Mac from ${esc(m.convert_from)} (a few minutes; that folder is left as it is)`
        : m.convert ? `${gb(m.size)} download, converted for this Mac; the download is deleted afterwards` : `${gb(m.size)} to download`}</div></div></div>`).join("");
  // Existing files: a ComfyUI folder (Windows, Linux), or on a Mac the official download or an earlier MLX conversion.
  const folderBox = (summary, placeholder, note) => `<details class="comfy" ${p.comfyui.length ? "open" : ""}><summary>${summary}</summary>
      <div class="row"><input id="comfyPath" placeholder="${placeholder}" value="${esc(p.comfyui[0] || "")}">
      <button class="btn small" id="comfyLook">Look there</button></div>
      <div class="muted small">${note}</div></details>`;
  const comfy = !toGet.length ? "" : mac
    ? folderBox("Already have the model on this Mac?", "A folder, e.g. ~/Downloads/Qwen-Image-2.1-Turbo",
        "Either the official Qwen-Image-2.1-Turbo download (it's converted from, and left as it is) or an MLX conversion made earlier (used as it is).")
    : folderBox("Already have these files?", "A folder with them, e.g. your ComfyUI folder C:\\ComfyUI",
        "Your ComfyUI folder, or any folder with the model files in it or in its subfolders. They're used where they are, not copied.");
  box.innerHTML = `<div class="setup-card">
    <h1>Set up Qwen Image Editor</h1>
    <p class="muted">A one-time setup. Everything runs on this computer.</p>
    <section><h2>This computer</h2>
      <div class="verdict ${mark}">${esc(hw.summary)}</div>
      <div class="muted">${facts}</div>
      ${hw.problems.map(t => `<div class="line bad">${esc(t)}</div>`).join("")}
      ${hw.warnings.map(t => `<div class="line note">${esc(t)}</div>`).join("")}
    </section>
    <section><h2>Model files</h2>
      ${models}
      ${comfy}
      ${toGet.length ? `<div class="muted small">Saved in ${esc(p.models_dir)}. Needs ${gb(p.disk_needed)} of free disk.</div>` : ""}
    </section>
    <section><h2>License</h2>
      <p>Qwen-Image-2.1 is released under the <a href="${esc(p.license.url)}" target="_blank" rel="noopener">${esc(p.license.name)}</a>:
      it may be used for research and other non-commercial purposes only. Commercial use needs a separate license from Qwen.</p>
      <label class="check"><input type="checkbox" id="acceptLicense" ${p.license.accepted ? "checked" : ""}> I accept the license</label>
    </section>
    ${chatAppsOffer(p.chat_apps)}
    ${S.status.setup_error ? `<p class="error">Setup stopped: ${esc(S.status.setup_error)}. Finished downloads are kept; try again.</p>` : ""}
    <div class="setup-actions">
      <button class="btn primary" id="setupGo" disabled>${action}</button>
      ${hw.verdict === "unsupported" ? `<label class="check small"><input type="checkbox" id="setupAnyway"> Try anyway</label>` : ""}
    </div>
  </div>`;
  const go = $("#setupGo");
  const update = () => { go.disabled = !$("#acceptLicense").checked || (hw.verdict === "unsupported" && !$("#setupAnyway").checked); };
  $("#acceptLicense").onchange = update;
  if ($("#setupAnyway")) $("#setupAnyway").onchange = update;
  update();
  if ($("#comfyLook")) $("#comfyLook").onclick = () => { const v = $("#comfyPath").value.trim(); if (v) loadSetupPlan([v]); };
  go.onclick = async () => {
    go.disabled = true;
    try {
      for (const box of $$("#setup input[data-chat-app]:checked")) await setChatApp(box.dataset.chatApp, true);
      await api.json("/api/setup/start", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accept_license: true, comfyui: p.comfyui }) });
      await refreshStatus();
    } catch (e) {
      $("#setup .setup-card").insertAdjacentHTML("beforeend", `<p class="error">${esc(e.message)}</p>`);
      go.disabled = false;
    }
  };
}

// ---------------------------------------------------------------- chat apps

// The editor can be used from Claude Desktop and Goose (its MCP server): connecting adds it to their settings.
const CHAT_APPS_HOW = "Then ask the chat to create or edit an image. In the picture that appears you can compare " +
  "results and mark up what to change.";

function chatAppsOffer(apps = []) {
  const offer = apps.filter(a => a.installed && !a.connected);
  if (!offer.length) return "";
  return `<section><h2>Chat apps</h2>
    <p>Also use the editor from ${offer.map(a => esc(a.name)).join(" and ")}: the AI there writes the prompts, and the
      images are made here.</p>
    ${offer.map(a => `<label class="check"><input type="checkbox" data-chat-app="${esc(a.key)}" checked> Connect to ${esc(a.name)}</label>`).join("")}
    <div class="muted small">Restart them afterwards. You can change this later with Chat apps, top right.</div>
  </section>`;
}

async function setChatApp(key, connect) {
  return api.json("/api/chat-apps", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key, connect }) });
}

async function renderChatApps(apps, justChanged) {
  const box = $("#chatApps");
  if (!apps) {
    box.innerHTML = `<p class="muted">Checking…</p>`;
    try { apps = await api.json("/api/chat-apps"); } catch (e) { box.innerHTML = `<p class="error">${esc(e.message)}</p>`; return; }
  }
  const changed = apps.find(a => a.key === justChanged);
  box.innerHTML = `<p>Use the editor from these apps: the AI there writes the prompts, and the images are made here.</p>` +
    apps.map(a => `<div class="chatapp"><span class="name">${esc(a.name)}</span>` +
      (a.connected ? `<span class="state on">Connected</span><button class="btn small ghost" data-key="${a.key}" data-on="0">Disconnect</button>`
       : a.installed ? `<button class="btn small" data-key="${a.key}" data-on="1">Connect</button>`
       : `<span class="state">Not installed</span>`) + `</div>`).join("") +
    (changed ? `<p class="note">${changed.connected ? "Connected" : "Disconnected"}. ${esc(changed.restart)}${changed.connected ? " " + CHAT_APPS_HOW : ""}</p>`
     : `<p class="muted small">${apps.some(a => a.connected) ? CHAT_APPS_HOW : "Connecting adds the editor to the app's settings. Restart the app afterwards."}</p>`);
  $$("button[data-key]", box).forEach(b => b.onclick = async () => {
    b.disabled = true;
    try { renderChatApps(await setChatApp(b.dataset.key, b.dataset.on === "1"), b.dataset.key); }
    catch (e) { box.insertAdjacentHTML("beforeend", `<p class="error">${esc(e.message)}</p>`); b.disabled = false; }
  });
}
$("#chatAppsBtn").onclick = () => {
  const box = $("#chatApps");
  togglePopover(box);
  if (!box.hidden) renderChatApps();
};

function renderSetupProgress() {
  const box = $("#setup"), st = S.status.setup || {};
  if (box.dataset.view !== "progress") {
    box.dataset.view = "progress";
    setupPlan = null;  // re-check after setup, in case it has to run again
    box.innerHTML = `<div class="setup-card">
      <h1>Setting up</h1>
      <p class="muted">This page can stay open or be closed: setup carries on, and the editor opens when it's done.</p>
      <div class="setup-label" id="setupLabel"></div>
      <div class="progress-track big"><div class="progress-fill" id="setupFill"></div></div>
      <div class="setup-meta"><span id="setupPct"></span><span id="setupRate"></span></div>
      <p class="note" id="setupTip" hidden></p>
    </div>`;
  }
  const f = st.fraction || 0;
  $("#setupLabel").textContent = st.label || "Starting";
  $("#setupFill").style.width = (f * 100).toFixed(1) + "%";
  $("#setupPct").textContent = Math.floor(f * 100) + "%";
  $("#setupRate").textContent = st.rate > 1e5 ? `${(st.rate / 1e6).toFixed(0)} MB/s` : "";
  $("#setupTip").hidden = !st.tip;
  $("#setupTip").textContent = st.tip || "";
}

// ---------------------------------------------------------------- composer

function updateComposer() {
  const n = S.refs.length, marks = S.refs.some(r => r.marks.length);
  $("#prompt").placeholder =
    n === 0 ? "Describe an image to create…" :
    marks ? "Describe the change. Refer to marks by colour, e.g. “remove the lamp in the red circle”." :
    n === 1 ? "Describe the change, e.g. “make it night time”. Click the image to mark areas." :
    `Describe how to combine them, e.g. “put the person from <image2> on the sofa in <image1>”.`;
  $("#addImage").innerHTML = icon("image") + "Add image";
  const sizeLabel = S.opts.size === "auto" ? "Auto size" : S.opts.size === "custom" ? `${S.opts.w} × ${S.opts.h}` : S.opts.size.replace("x", " × ");
  $("#optionsBtn").innerHTML = icon("sliders") + sizeLabel + (S.opts.seed !== "" ? ` · seed ${S.opts.seed}` : "");
  $("#composerHint").textContent = navigator.platform.includes("Mac") ? "⌘ Enter" : "Ctrl Enter";
  updateGenerateButton();
}
function updateGenerateButton() {
  const b = $("#generate");
  b.disabled = !["ready", "idle"].includes(S.status.state) || !!S.job || !$("#prompt").value.trim();
  if (!S.job) b.textContent = "Generate";
}

function renderAttachments() {
  const box = $("#attachments");
  box.innerHTML = "";
  S.refs.forEach((ref, i) => {
    const el = document.createElement("div");
    el.className = "att" + (S.mode === "ref" && S.refIndex === i ? " active" : "");
    el.draggable = true;
    el.title = `Image ${i + 1}: click to mark areas, drag to reorder`;
    el.innerHTML = `${markedImage(ref)}<span class="num">${i + 1}</span>` +
      (ref.marks.length ? `<span class="marks">${ref.marks.map(m => `<span class="swatch" style="background:${m.color.css}"></span>`).join("")}</span>` : "") +
      `<button class="remove" title="Remove">${icon("x")}</button>`;
    el.onclick = () => openRef(i);
    $(".remove", el).onclick = e => { e.stopPropagation(); removeRef(i); };
    el.ondragstart = e => { e.dataTransfer.setData("text/x-ref", String(i)); el.classList.add("dragging"); };
    el.ondragend = () => el.classList.remove("dragging");
    el.ondragover = e => { if (e.dataTransfer.types.includes("text/x-ref")) e.preventDefault(); };
    el.ondrop = e => {
      const from = +e.dataTransfer.getData("text/x-ref");
      if (Number.isNaN(from) || from === i) return;
      e.preventDefault(); e.stopPropagation();
      const [moved] = S.refs.splice(from, 1);
      S.refs.splice(i, 0, moved);
      if (S.mode === "ref") S.refIndex = S.refs.indexOf(moved);
      refsChanged();
    };
    box.appendChild(el);
  });
  updateComposer();
}
function refsChanged() {
  renderAttachments();
  if (S.mode === "ref") renderCanvas();
}
function removeRef(i) {
  S.refs.splice(i, 1);
  if (S.mode === "ref") {
    if (S.refIndex === i) return closeRef();
    if (S.refIndex > i) S.refIndex--;
  }
  refsChanged();
}

async function addFiles(files) {
  hideError();
  let last = null;
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    try { last = await addRef(f); } catch (e) { showError(e.message); }
  }
  renderAttachments();
  // Show what was just added unless a result is being looked at.
  if (last && S.mode !== "result") openRef(S.refs.indexOf(last));
}

// ---------------------------------------------------------------- options popover

function setupOptions() {
  const o = S.opts;
  $("#optSize").value = o.size;
  $("#optW").value = o.w; $("#optH").value = o.h;
  $("#optSeed").value = o.seed;
  $("#optKv").checked = o.kv;
  $("#optHideMarks").checked = o.hideMarks;
  const sync = () => {
    o.size = $("#optSize").value;
    o.w = +$("#optW").value || 1024; o.h = +$("#optH").value || 1024;
    o.seed = $("#optSeed").value.trim();
    o.kv = $("#optKv").checked;
    o.hideMarks = $("#optHideMarks").checked;
    $("#optCustom").hidden = o.size !== "custom";
    const size = chosenSize();
    $("#optSizeWarn").hidden = !(size && size[0] * size[1] > 1.3e6);
    saveOpts();
    updateComposer();
  };
  $$("#options select, #options input").forEach(el => el.addEventListener("input", sync));
  sync();
  $("#optionsBtn").onclick = e => { e.stopPropagation(); togglePopover($("#options")); };
}
function chosenSize() {
  const o = S.opts;
  if (o.size === "auto") return null;
  if (o.size === "custom") return [o.w, o.h];
  return o.size.split("x").map(Number);
}
function togglePopover(el, show) {
  const open = show ?? el.hidden;
  $$(".popover").forEach(p => { if (p !== el) p.hidden = true; });
  el.hidden = !open;
}
document.addEventListener("click", e => { if (!e.target.closest(".popover, .popover-anchor")) $$(".popover").forEach(p => p.hidden = true); });

// ---------------------------------------------------------------- marks

const strokeWidth = ref => Math.max(3, Math.round(Math.max(ref.w, ref.h) / 190));  // visible to the model at any size
function nextColor() {
  const used = new Set(S.refs.flatMap(r => r.marks.map(m => m.color.name)));
  const count = S.refs.reduce((n, r) => n + r.marks.length, 0);
  return MARK_COLORS.find(c => !used.has(c.name)) || MARK_COLORS[count % MARK_COLORS.length];
}
const pathOf = m => m.points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ") + (m.closed ? " Z" : "");
const markPhrase = m => m.closed ? `in the ${m.color.name} circle` : `along the ${m.color.name} line`;
function markedImage(ref) {
  const sw = strokeWidth(ref);
  return `<span class="marked"><img src="${refUrl(ref.id)}" alt="">` +
    `<svg viewBox="0 0 ${ref.w} ${ref.h}" preserveAspectRatio="none">` +
    ref.marks.map(m => `<path d="${pathOf(m)}" fill="none" stroke="${m.color.css}" stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round"/>`).join("") +
    `</svg></span>`;
}

function bindDrawing(svg, ref) {
  let live = null;
  const toXY = e => {
    const b = svg.getBoundingClientRect();
    return [(e.clientX - b.left) * ref.w / b.width, (e.clientY - b.top) * ref.h / b.height];
  };
  svg.onpointerdown = e => {
    if (e.button !== 0) return;
    svg.setPointerCapture(e.pointerId);
    live = { color: nextColor(), points: [toXY(e)], closed: false };
    live.el = document.createElementNS("http://www.w3.org/2000/svg", "path");
    Object.entries({ fill: "none", stroke: live.color.css, "stroke-width": strokeWidth(ref), "stroke-linejoin": "round", "stroke-linecap": "round" })
      .forEach(([k, v]) => live.el.setAttribute(k, v));
    svg.appendChild(live.el);
  };
  svg.onpointermove = e => {
    if (!live) return;
    const [x, y] = toXY(e), [px, py] = live.points[live.points.length - 1];
    if (Math.hypot(x - px, y - py) < Math.max(ref.w, ref.h) / 400) return;
    live.points.push([x, y]);
    live.el.setAttribute("d", pathOf(live));
  };
  svg.onpointerup = svg.onpointercancel = () => {
    if (!live) return;
    const m = live; live = null;
    m.el.remove(); delete m.el;
    const xs = m.points.map(p => p[0]), ys = m.points.map(p => p[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const size = Math.max(ref.w, ref.h);
    if (m.points.length < 3 || span < size / 80) return renderCanvas();  // a click, not a mark
    const [x0, y0] = m.points[0], [x1, y1] = m.points[m.points.length - 1];
    m.closed = Math.hypot(x1 - x0, y1 - y0) < Math.max(span * 0.35, size / 25);  // the ends meet: a loop
    ref.marks.push(m);
    refsChanged();
  };
}

// Draw the marks into the pixels, the way the model will see them, and upload that copy.
async function flatten(ref) {
  const img = await loadImage(refUrl(ref.id));
  const c = document.createElement("canvas");
  c.width = ref.w; c.height = ref.h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = strokeWidth(ref);
  ctx.lineJoin = ctx.lineCap = "round";
  for (const m of ref.marks) { ctx.strokeStyle = m.color.css; ctx.stroke(new Path2D(pathOf(m))); }
  const blob = await new Promise(ok => c.toBlob(ok, "image/png"));
  return (await api.upload(blob)).id;
}

function insertIntoPrompt(text) {
  const t = $("#prompt");
  const a = t.selectionStart ?? t.value.length, b = t.selectionEnd ?? a;
  const before = t.value.slice(0, a), after = t.value.slice(b);
  const pre = before && !/\s$/.test(before) ? " " : "", post = after && !/^[\s,.;]/.test(after) ? " " : "";
  t.value = before + pre + text + post + after;
  const pos = (before + pre + text + post).length;
  t.focus(); t.setSelectionRange(pos, pos);
  autosize(); updateGenerateButton();
}

// ---------------------------------------------------------------- canvas

function openRef(i) {
  S.mode = "ref"; S.refIndex = i;
  renderAttachments(); renderCanvas();
}
function closeRef() {
  S.refIndex = null;
  S.mode = S.current ? "result" : "empty";
  renderAttachments(); renderCanvas(); renderHistory();
}
function showResult(rec, fresh = false) {
  S.current = rec; S.mode = "result"; S.refIndex = null;
  const targets = compareTargets(rec);
  if (!targets.some(t => t.key === S.against)) {
    const src = sourcesOf(rec);
    S.against = src.length ? "ref:" + src[src.length - 1].id : targets[0]?.key ?? null;
  }
  if (!targets.length) S.view = "result";
  renderAttachments(); renderCanvas(fresh); renderHistory();
}

const sourcesOf = rec => rec.sources || rec.refs.map(id => ({ id, marks: [] }));
function compareTargets(rec) {
  const t = sourcesOf(rec).map((s, i) => ({ key: "ref:" + s.id, label: `Image ${i + 1}`, url: refUrl(s.id), w: s.w, h: s.h }));
  const idx = S.history.findIndex(h => h.id === rec.id);
  const prev = S.history.slice(idx + 1, idx + 6);
  prev.forEach((h, k) => t.push({ key: "res:" + h.id, label: k === 0 ? "Previous result" : `Result · ${h.prompt.slice(0, 24)}…`, url: resultUrl(h), w: h.width, h: h.height }));
  return t;
}

function frame(inner, w, h, cls = "") {
  return `<div class="frame ${cls}" data-w="${w}" data-h="${h}">${inner}</div>`;
}
// Size each frame to fit its container while keeping the image's aspect ratio (CSS can't do this for
// nested auto-height boxes).
function fitFrames() {
  $$("#canvasBody .frame").forEach(f => {
    const box = f.parentElement.getBoundingClientRect();
    const w = +f.dataset.w, h = +f.dataset.h;
    const scale = Math.min(box.width / w, box.height / h, 2);
    f.style.width = Math.floor(w * scale) + "px";
    f.style.height = Math.floor(h * scale) + "px";
  });
}
new ResizeObserver(fitFrames).observe(document.getElementById("canvasBody"));

function renderCanvas(fresh = false) {
  const body = $("#canvasBody"), bar = $("#floatbar");
  body.className = "canvas-body";
  if (S.mode === "ref" && S.refs[S.refIndex]) {
    const ref = S.refs[S.refIndex];
    body.innerHTML = frame(markedImage(ref), ref.w, ref.h, "drawing");
    $$(".frame .marked, .frame .marked img", body).forEach(el => { el.style.width = "100%"; el.style.height = "100%"; });
    bindDrawing($(".frame svg", body), ref);
    renderMarkBar(ref);
  } else if (S.mode === "result" && S.current) {
    renderResult(fresh);
    renderResultBar();
  } else {
    bar.hidden = true;
    body.classList.add("no-bar");
    body.innerHTML = `<div class="empty"><div class="drop" id="emptyDrop">${icon("upload")}
      <h2>Drop an image to edit it</h2><p>or paste one, or describe a new image below.</p>
      <button class="btn">${icon("image")}Choose image</button></div></div>`;
    $("#emptyDrop").onclick = () => $("#file").click();
  }
  fitFrames();
}

function renderMarkBar(ref) {
  const bar = $("#floatbar");
  bar.hidden = false;
  const chips = ref.marks.map((m, k) =>
    `<span class="chip"><span class="swatch" style="background:${m.color.css}"></span>` +
    `<button class="name" data-k="${k}" title="Insert “${markPhrase(m)}” into the prompt">${m.color.name}</button>` +
    `<button class="del" data-k="${k}" title="Delete this mark">${icon("x")}</button></span>`).join("");
  bar.innerHTML =
    `<span class="label">${icon("pen")}Image ${S.refIndex + 1} · ${ref.marks.length ? "Click a colour to use it in the prompt" : "Draw around what should change"}</span>` +
    chips +
    (ref.marks.length ? `<span class="sep"></span>
      <button class="btn small icon-only ghost" id="markUndo" title="Undo (Ctrl+Z)">${icon("undo")}</button>
      <button class="btn small icon-only ghost" id="markClear" title="Clear all marks">${icon("trash")}</button>` : "") +
    `<span class="sep"></span><button class="btn small" id="markDone">${icon("check")}Done</button>`;
  $$(".chip .name", bar).forEach(b => b.onclick = () => insertIntoPrompt(markPhrase(ref.marks[+b.dataset.k])));
  $$(".chip .del", bar).forEach(b => b.onclick = () => { ref.marks.splice(+b.dataset.k, 1); refsChanged(); });
  if (ref.marks.length) {
    $("#markUndo").onclick = () => { ref.marks.pop(); refsChanged(); };
    $("#markClear").onclick = () => { ref.marks = []; refsChanged(); };
  }
  $("#markDone").onclick = closeRef;
}

function renderResult(fresh) {
  const body = $("#canvasBody"), rec = S.current;
  const target = compareTargets(rec).find(t => t.key === S.against);
  const view = target ? S.view : "result";
  const a = resultUrl(rec), w = rec.width, h = rec.height, anim = fresh ? "fade-in" : "";
  if (view === "result") {
    body.innerHTML = frame(`<img src="${a}" alt="">`, w, h, anim);
  } else if (view === "side") {
    // Each image in its own shape: the before image can have a different size or aspect ratio.
    body.innerHTML = `<div class="side">
      <div class="cell">${frame(`<img src="${target.url}" alt=""><span class="tag">${esc(target.label)}</span>`, target.w || w, target.h || h)}</div>
      <div class="cell">${frame(`<img src="${a}" alt=""><span class="tag">Result</span>`, w, h, anim)}</div></div>`;
    if (!target.w) {  // older records don't carry the reference's size
      loadImage(target.url).then(img => {
        const f = $(".side .cell .frame", body);
        if (f) { f.dataset.w = img.naturalWidth; f.dataset.h = img.naturalHeight; fitFrames(); }
      });
    }
  } else if (view === "compare") {
    // The slider keeps the result's frame; a before image of another shape is fitted inside it, not stretched.
    body.innerHTML = frame(`<div class="compare"><img src="${a}" alt=""><img class="over" src="${target.url}" alt="">
      <div class="handle"></div><span class="tag">${esc(target.label)}</span><span class="tag right">Result</span></div>`, w, h, anim);
    const cmp = $(".compare", body), over = $(".over", cmp), handle = $(".handle", cmp);
    const set = p => { over.style.clipPath = `inset(0 ${100 - p * 100}% 0 0)`; handle.style.left = p * 100 + "%"; };
    set(0.5);
    cmp.onpointermove = e => { const r = cmp.getBoundingClientRect(); set(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))); };
  } else if (view === "diff") {
    const reshaped = target.w && Math.abs(target.w / target.h - w / h) > 0.02;
    body.innerHTML = frame(`<canvas></canvas><span class="tag">Red shows what changed${reshaped ? " · the images have different shapes" : ""}</span>`, w, h, anim);
    drawDiff(a, target.url, $("canvas", body));
  }
  $$(".frame img, .frame canvas, .compare", body).forEach(el => { if (!el.classList.contains("over")) { el.style.width = "100%"; el.style.height = "100%"; } });
}

function renderResultBar() {
  const bar = $("#floatbar"), rec = S.current;
  const targets = compareTargets(rec);
  bar.hidden = false;
  const views = [["result", "Result"], ["compare", "Compare"], ["side", "Side by side"], ["diff", "Changes"]];
  bar.innerHTML =
    `<div class="seg">${views.map(([k, l]) => `<button data-view="${k}" class="${S.view === k ? "on" : ""}" ${k !== "result" && !targets.length ? "disabled" : ""}>${l}</button>`).join("")}</div>` +
    (targets.length && S.view !== "result" ? `<select id="against" title="Compare with">${targets.map(t => `<option value="${t.key}" ${t.key === S.against ? "selected" : ""}>vs ${esc(t.label)}</option>`).join("")}</select>` : "") +
    `<span class="sep"></span>
    <button class="btn small" id="actEdit" title="Use this result as the image to edit next">${icon("edit")}Edit this</button>
    <button class="btn small ghost" id="actAddRef" title="Add this result to the reference images">${icon("layers")}Add as reference</button>
    <button class="btn small ghost" id="actRestore" title="Put this prompt, its images and settings back into the composer">${icon("restore")}Reuse prompt</button>
    <span class="sep"></span>
    <a class="btn small icon-only ghost" id="actDownload" title="Download" href="${resultUrl(rec)}" download="qwen-${rec.id}.png">${icon("download")}</a>
    <div class="popover-anchor"><button class="btn small icon-only ghost" id="actInfo" title="Details">${icon("info")}</button>
      <div class="popover info" id="infoPop" hidden>${infoHtml(rec)}</div></div>`;
  $$(".seg button", bar).forEach(b => b.onclick = () => { S.view = b.dataset.view; renderCanvas(); });
  const against = $("#against");
  if (against) against.onchange = () => { S.against = against.value; renderCanvas(); };
  $("#actEdit").onclick = async () => {
    const blob = await (await fetch(resultUrl(rec))).blob();
    S.refs = [];
    await addRef(blob);
    $("#prompt").value = ""; autosize();
    S.opts.size = "auto"; setupOptions();
    openRef(0);
    $("#prompt").focus();
  };
  $("#actAddRef").onclick = async () => {
    const blob = await (await fetch(resultUrl(rec))).blob();
    await addRef(blob);
    renderAttachments();
  };
  $("#actRestore").onclick = () => restoreInputs(rec);
  $("#actInfo").onclick = e => { e.stopPropagation(); togglePopover($("#infoPop")); };
}

function infoHtml(rec) {
  const st = rec.stages_s || {}, row = (k, v) => v == null ? "" : `<dt>${k}</dt><dd>${v}</dd>`;
  const sec = v => v == null ? null : v.toFixed(2) + " s";
  return `<p class="prompt-full">${esc(rec.prompt)}</p><dl>` +
    row("Size", `${rec.width} × ${rec.height}`) + row("Seed", rec.seed) + row("Total", sec(rec.total_s)) +
    row("Reading the prompt", sec(st.text_encoder)) + row("Encoding references", sec(st.vae_encode)) +
    row("8 steps", sec(st.transformer)) + row("Decoding", sec(st.vae_decode)) +
    row("Peak VRAM", rec.peak_reserved_gib + " GiB") +
    row("Blocks streamed", rec.plan && rec.plan.streamed_blocks != null ? rec.plan.streamed_blocks : null) +
    row("Reference cache", rec.kv === false ? "off" : "on") + `</dl>`;
}

function restoreInputs(rec) {
  $("#prompt").value = rec.prompt.replace(" " + NO_MARKS, "");
  autosize();
  S.refs = sourcesOf(rec).map(s => ({
    id: s.id, w: s.w, h: s.h,
    marks: (s.marks || []).map(m => ({ ...m, color: MARK_COLORS.find(c => c.name === m.color) || MARK_COLORS[0] })),
  }));
  // Records from before marks existed don't carry sizes; read them from the image.
  S.refs.filter(r => !r.w).forEach(r => loadImage(refUrl(r.id)).then(img => { r.w = img.naturalWidth; r.h = img.naturalHeight; renderAttachments(); }));
  if ($("#optSize").querySelector(`option[value="${rec.size_mode}"]`)) S.opts.size = rec.size_mode;
  if (rec.size_mode === "custom") { S.opts.w = rec.width; S.opts.h = rec.height; }
  S.opts.kv = rec.kv !== false;
  setupOptions();
  renderAttachments();
}

const loadImage = url => new Promise((ok, err) => { const i = new Image(); i.onload = () => ok(i); i.onerror = err; i.src = url; });
async function drawDiff(aUrl, bUrl, canvas) {
  const [a, b] = await Promise.all([loadImage(aUrl), loadImage(bUrl)]);
  const w = a.naturalWidth, h = a.naturalHeight;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  // The before image fitted into the result's frame, keeping its aspect ratio (as on the slider). Where it doesn't
  // reach, there's nothing to compare: those pixels stay grey.
  const s = Math.min(w / b.naturalWidth, h / b.naturalHeight);
  const bw = Math.round(b.naturalWidth * s), bh = Math.round(b.naturalHeight * s);
  const bx = Math.floor((w - bw) / 2), by = Math.floor((h - bh) / 2);
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(b, bx, by, bw, bh);
  const pb = ctx.getImageData(0, 0, w, h).data;
  ctx.drawImage(a, 0, 0, w, h);
  const out = ctx.getImageData(0, 0, w, h), pa = out.data;
  for (let i = 0; i < pa.length; i += 4) {
    const d = (Math.abs(pa[i] - pb[i]) + Math.abs(pa[i + 1] - pb[i + 1]) + Math.abs(pa[i + 2] - pb[i + 2])) / 3;
    const g = (pa[i] * 0.3 + pa[i + 1] * 0.59 + pa[i + 2] * 0.11) * 0.4;
    const heat = pb[i + 3] ? Math.min(1, Math.max(0, (d - 12) / 60)) : 0;
    pa[i] = g + (255 - g) * heat; pa[i + 1] = g * (1 - heat * 0.8); pa[i + 2] = g * (1 - heat * 0.8); pa[i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

// ---------------------------------------------------------------- generate

const STAGES = [["text", "Prompt"], ["refs", "References"], ["steps", "Steps"], ["decode", "Decode"]];
const STAGE_TITLES = { queued: "Starting", text: "Reading the prompt", refs: "Encoding the references", steps: "Generating", decode: "Decoding" };

// Expected seconds per stage: the last comparable run if there is one, else what an RTX 3060 measures.
function expectedStages(nRefs) {
  const prev = S.history.find(h => (h.refs || []).length === nRefs && h.stages_s);
  const st = prev ? prev.stages_s : {};
  return {
    text: st.text_encoder ?? (nRefs ? 1.2 * nRefs : 0.45),
    refs: nRefs ? st.vae_encode ?? 0.36 * nRefs : 0,
    steps: st.transformer ?? 10.6 + 3.4 * nRefs,
    decode: st.vae_decode ?? 1.45,
  };
}

async function generate() {
  if ($("#generate").disabled) return;
  hideError();
  let prompt = $("#prompt").value.trim();
  const anyMarks = S.refs.some(r => r.marks.length);
  if (anyMarks && S.opts.hideMarks && !prompt.includes(NO_MARKS)) prompt = prompt.replace(/[.\s]*$/, ".") + " " + NO_MARKS;
  const size = chosenSize();
  const body = {
    prompt,
    size_mode: S.opts.size,
    width: size ? size[0] : null,
    height: size ? size[1] : null,
    seed: S.opts.seed === "" ? null : S.opts.seed,
    kv: S.opts.kv,
    sources: S.refs.map(r => ({ id: r.id, w: r.w, h: r.h, marks: r.marks.map(m => ({ color: m.color.name, points: m.points, closed: m.closed })) })),
  };
  const exp = expectedStages(S.refs.length);
  S.job = { t0: performance.now(), exp, total: Object.values(exp).reduce((a, b) => a + b, 0), nRefs: S.refs.length, prompt, pct: 0, stage: "queued", step: 0, steps: 8 };
  S.job.poll = setInterval(async () => { try { S.status = await api.status(); } catch {} }, 200);
  showWorking();
  $("#progress").hidden = false;
  renderHistory();
  tickProgress();
  try {
    body.refs = await Promise.all(S.refs.map(r => r.marks.length ? flatten(r) : r.id));
    const rec = await api.generate(body);
    S.history.unshift(rec);
    S.job = stopJob();
    S.view = "result";  // a new result is shown as itself first
    showResult(rec, true);
  } catch (e) {
    S.job = stopJob();
    showError(e.message);
    S.mode = S.current ? "result" : "empty";
    renderCanvas();
    renderHistory();
  }
  refreshStatus();
}
// While generating, the canvas shows what's being worked on (the last reference, marks included) behind the progress.
function showWorking() {
  S.mode = "working"; S.refIndex = null;
  const ref = S.refs[S.refs.length - 1];
  $("#floatbar").hidden = true;
  $("#canvasBody").className = "canvas-body no-bar";
  $("#canvasBody").innerHTML = ref ? frame(markedImage(ref), ref.w, ref.h) : "";
  $$("#canvasBody .marked, #canvasBody .marked img").forEach(el => { el.style.width = "100%"; el.style.height = "100%"; });
  fitFrames();
  renderAttachments();
}
function stopJob() {
  if (S.job) clearInterval(S.job.poll);
  $("#progress").hidden = true;
  return null;
}

// Progress moves with time between the stage boundaries the server reports, so it never jumps back or stalls.
function tickProgress() {
  const job = S.job;
  if (!job) return;
  const s = S.status;
  if (s.busy && s.stage) { job.stage = s.stage; job.step = s.step || 0; job.steps = s.steps || job.steps; }
  const elapsed = (performance.now() - job.t0) / 1000;
  const order = ["text", "refs", "steps", "decode"], idx = order.indexOf(job.stage), e = job.exp;
  let floor = 0;
  for (let k = 0; k < Math.max(idx, 0); k++) floor += e[order[k]];
  let ceil = floor + (idx >= 0 ? e[order[idx]] : 0);
  if (job.stage === "steps") { floor += e.steps * job.step / job.steps; ceil = floor + e.steps / job.steps; }
  const t = Math.min(Math.max(elapsed, floor), ceil - 0.02 * (ceil - floor));
  job.pct = Math.max(job.pct, Math.min(0.99, t / job.total));

  $("#progTitle").textContent = s.state === "loading" ? `Loading the model · ${s.load_pct || 0}%`
    : STAGE_TITLES[job.stage] + (job.stage === "steps" ? ` · step ${Math.min(job.step + 1, job.steps)} of ${job.steps}` : "");
  $("#progTime").textContent = `${elapsed.toFixed(1)} s`;
  $("#progFill").style.width = (job.pct * 100).toFixed(1) + "%";
  $("#progStages").innerHTML = STAGES.filter(([k]) => k !== "refs" || job.nRefs).map(([k, label]) => {
    const i = order.indexOf(k);
    return `<span class="${idx > i ? "done" : idx === i ? "now" : ""}">${label}</span>`;
  }).join("");
  $("#generate").textContent = `${Math.round(job.pct * 100)}%`;
  $("#generate").disabled = true;
  const pending = $("#historyList .pending .meta");
  if (pending) pending.textContent = `${Math.round(job.pct * 100)}% · ${elapsed.toFixed(0)} s`;
  setTimeout(tickProgress, 120);
}

function showError(msg) { const e = $("#error"); e.textContent = msg; e.hidden = false; }
function hideError() { $("#error").hidden = true; }

// ---------------------------------------------------------------- history

function renderHistory() {
  const list = $("#historyList");
  const ago = rec => {
    const m = /^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)(\d\d)/.exec(rec.id);
    if (!m) return "";
    const d = new Date(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
    const s = (Date.now() - d) / 1000;
    return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : d.toLocaleDateString();
  };
  const pending = S.job ? `<div class="card pending"><div class="thumb"></div><div class="body">
      <div class="text">${esc(S.job.prompt.replace(" " + NO_MARKS, ""))}</div><div class="meta">Starting…</div></div></div>` : "";
  list.innerHTML = pending + (S.history.length ? S.history.map(rec =>
    `<div class="card ${S.mode === "result" && S.current && S.current.id === rec.id ? "active" : ""}" data-id="${rec.id}">
      <button class="card-del" title="Delete">${icon("trash")}</button>
      <div class="thumb"><img src="${resultUrl(rec)}" alt="" loading="lazy"></div>
      <div class="body"><div class="text">${esc(rec.prompt.replace(" " + NO_MARKS, ""))}</div>
      <div class="meta">${ago(rec)} · ${rec.total_s.toFixed(1)} s${rec.refs.length ? ` · ${rec.refs.length} image${rec.refs.length > 1 ? "s" : ""}` : ""}</div></div></div>`).join("")
    : (S.job ? "" : `<div class="none">Your results will appear here.</div>`));
  $$(".card[data-id]", list).forEach(c => c.onclick = () => showResult(S.history.find(h => h.id === c.dataset.id)));
  $$(".card .card-del", list).forEach(b => b.onclick = e => { e.stopPropagation(); deleteHistory([b.closest(".card").dataset.id]); });
  $("#historyCount").textContent = S.history.length || "";
  renderHistoryActions();
}

// "Clear all" asks inline (no browser dialog) and calms down again after a few seconds.
let confirmTimer = null;
function renderHistoryActions(confirming = false) {
  const box = $("#historyActions");
  clearTimeout(confirmTimer);
  if (!S.history.length || S.job) { box.innerHTML = ""; return; }
  if (!confirming) {
    box.innerHTML = `<button class="btn small ghost" id="clearAll">Clear all</button>`;
    $("#clearAll").onclick = () => renderHistoryActions(true);
    return;
  }
  box.innerHTML = `<span class="confirm">Delete all ${S.history.length}?
    <button class="btn small danger" id="clearYes">Delete</button><button class="btn small ghost" id="clearNo">Cancel</button></span>`;
  $("#clearYes").onclick = () => deleteHistory(null);
  $("#clearNo").onclick = () => renderHistoryActions(false);
  confirmTimer = setTimeout(() => renderHistoryActions(false), 6000);
}

// ids = null deletes everything. Uploads still attached in the composer are kept.
async function deleteHistory(ids) {
  try {
    const keep = S.refs.map(r => r.id);
    const { deleted } = await api.deleteHistory(ids ? { ids, keep } : { all: true, keep });
    S.history = S.history.filter(h => !deleted.includes(h.id));
    if (S.current && deleted.includes(S.current.id)) {
      S.current = null;
      if (S.mode === "result") S.mode = "empty";
      renderCanvas();
    } else if (S.mode === "result") {
      renderCanvas();  // compare targets may have changed
    }
    renderHistory();
  } catch (e) { showError(e.message); }
}

// ---------------------------------------------------------------- input wiring

function autosize() {
  const t = $("#prompt");
  t.style.height = "auto";
  t.style.height = Math.min(180, t.scrollHeight) + "px";
}
$("#prompt").addEventListener("input", () => { autosize(); updateGenerateButton(); });
$("#generate").onclick = generate;
$("#quit").onclick = async () => {
  stopped = true;
  try { await api.json("/api/quit", { method: "POST" }); } catch {}
  document.body.innerHTML = `<div class="stopped"><h1>Qwen Image Editor has stopped</h1>
    <p>You can close this tab. To start it again, open Qwen Image Editor from the Start menu or Applications.</p></div>`;
};
$("#addImage").onclick = () => $("#file").click();
$("#file").onchange = e => { addFiles([...e.target.files]); e.target.value = ""; };

document.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); return; }
  if (e.key === "Escape") {
    if ($$(".popover").some(p => !p.hidden)) return $$(".popover").forEach(p => p.hidden = true);
    if (S.mode === "ref") closeRef();
    else if (S.mode === "result" && S.view !== "result") { S.view = "result"; renderCanvas(); }  // leave compare views
    return;
  }
  const typing = /^(TEXTAREA|INPUT|SELECT)$/.test(e.target.tagName);
  if (!typing && S.mode === "ref" && e.key.toLowerCase() === "z" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    const ref = S.refs[S.refIndex];
    if (ref && ref.marks.length) { ref.marks.pop(); refsChanged(); }
  }
});

// Files dropped anywhere become references; dragging a thumbnail to reorder doesn't trigger the veil.
let dragDepth = 0;
const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
document.addEventListener("dragenter", e => { if (hasFiles(e)) { dragDepth++; $("#dropveil").classList.add("show"); } });
document.addEventListener("dragleave", e => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $("#dropveil").classList.remove("show"); } });
document.addEventListener("dragover", e => { if (hasFiles(e)) e.preventDefault(); });
document.addEventListener("drop", e => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("#dropveil").classList.remove("show");
  addFiles([...e.dataTransfer.files]);
});
document.addEventListener("paste", e => {
  const files = [...e.clipboardData.items].filter(i => i.kind === "file").map(i => i.getAsFile()).filter(Boolean);
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ---------------------------------------------------------------- start

(async () => {
  setupOptions();
  renderAttachments();
  renderCanvas();
  try { S.history = await api.history(); } catch {}
  renderHistory();
  pollStatus();
})();
