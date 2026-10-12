(() => {
"use strict";

const data = window.RAG_COOKBOOK_DATA || { categories: [], components: [] };
const $ = (s) => document.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

/* Filtering is a different question from searching: search is "does this match
   what I typed", filter is "which of these can I actually use right now" --
   a component behind an optional package or a network service is not something
   you can drop in without setup, and that is the first thing to rule out. */
const FILTERS = [
  ["all", "All components"],
  ["bare", "No extra packages"],
  ["optional", "Optional packages available"],
  ["service", "Needs an external service"],
];

let query = "";
let filter = "all";
let active = null;
const collapsed = new Set();

/* ------------------------------------------------------------------ routing */

/* Hash routing rather than in-memory state: opened straight from disk there is
   no server to route on, and a URL that does not reflect what you are looking
   at means the back button cannot undo a click and a component cannot be sent
   to yourself. */
function go(path) {
  location.hash = path ? "#/" + path : "#/";
}

function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (!parts.length) return home();
  if (parts.length === 1) return section(parts[0]);
  return detail(parts[0], parts[1]);
}

/* -------------------------------------------------------------- markdown */

/* A small subset renderer -- headings, fences, tables, lists, bold, code and
   links -- covering what the component READMEs actually use. The text is
   escaped first and inline rules run on the escaped string, so a component
   documenting an `<int>` or an `&` cannot inject markup into the page. */
function md(src) {
  if (!src || !src.trim())
    return '<p class="muted">Not recorded in this component\'s documentation.</p>';

  const inline = (t) =>
    esc(t)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

  const lines = String(src).split("\n");
  const out = [];
  let list = null;
  let i = 0;

  const closeList = () => {
    if (list) {
      out.push("</" + list + ">");
      list = null;
    }
  };
  const startsBlock = (l) =>
    /^(```|#{2,5}\s|\s*\||\s*[-*]\s|\s*\d+[.)]\s)/.test(l);

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith("```")) {
      closeList();
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) buf.push(lines[i++]);
      i++;
      out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
      continue;
    }

    const h = /^(#{2,5})\s+(.*)$/.exec(line);
    if (h) {
      closeList();
      const level = h[1].length;
      out.push("<h" + level + ">" + inline(h[2]) + "</h" + level + ">");
      i++;
      continue;
    }

    /* A table is a row of cells followed by a separator row of dashes/colons. */
    if (/^\s*\|/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
      closeList();
      const cells = (l) =>
        l.trim().replace(/^\||\|$/g, "").split("|").map((s) => s.trim());
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(
        "<table><thead><tr>" +
          head.map((c) => "<th>" + inline(c) + "</th>").join("") +
          "</tr></thead><tbody>" +
          rows
            .map((r) => "<tr>" + r.map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>")
            .join("") +
          "</tbody></table>"
      );
      continue;
    }

    const li = /^\s*[-*]\s+(.*)$/.exec(line);
    if (li) {
      if (list !== "ul") {
        closeList();
        out.push("<ul>");
        list = "ul";
      }
      out.push("<li>" + inline(li[1]) + "</li>");
      i++;
      continue;
    }

    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (ol) {
      if (list !== "ol") {
        closeList();
        out.push("<ol>");
        list = "ol";
      }
      out.push("<li>" + inline(ol[1]) + "</li>");
      i++;
      continue;
    }

    if (!line.trim()) {
      closeList();
      i++;
      continue;
    }

    closeList();
    const buf = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) buf.push(lines[i++]);
    out.push("<p>" + inline(buf.join(" ")) + "</p>");
  }
  closeList();
  return out.join("\n");
}

/* ------------------------------------------------------------------ helpers */

const categoryOf = (id) => data.categories.find((c) => c.id === id);
const componentOf = (id) => data.components.find((c) => c.id === id);
const catName = (id) => categoryOf(id)?.name || id;

function matches(c) {
  const q = query.trim().toLowerCase();
  if (q) {
    const haystack = [c.name, c.id, c.description, c.purpose, catName(c.category)]
      .join(" ")
      .toLowerCase();
    if (!haystack.includes(q)) return false;
  }
  if (filter === "bare")
    return (
      !c.required_packages.length &&
      !c.optional_packages.length &&
      !c.external_services.length
    );
  if (filter === "optional") return c.optional_packages.length > 0;
  if (filter === "service") return c.external_services.length > 0;
  return true;
}

function crumbs(trail) {
  $("#crumb").innerHTML = ['<a href="#/">Home</a>']
    .concat(trail.map((t) => (t.href ? '<a href="' + esc(t.href) + '">' + esc(t.label) + "</a>" : "<span>" + esc(t.label) + "</span>")))
    .join(' <span class="sep">/</span> ');
}

function card(c) {
  return (
    '<button class="card" data-goto="' + esc(c.category + "/" + c.id) + '">' +
    "<strong>" + esc(c.name) + "</strong>" +
    '<span class="muted">' + esc(c.description) + "</span>" +
    '<span class="meta">' + statusTag(c) + fileTag(c) + "</span></button>"
  );
}

const statusTag = (c) =>
  '<span class="tag ' + (c.status === "stable" ? "tag-good" : "tag-warn") + '">' +
  esc(c.status) + "</span>";

const fileTag = (c) =>
  c.external_services.length
    ? '<span class="tag tag-svc">external service</span>'
    : c.optional_packages.length
    ? '<span class="tag">optional packages</span>'
    : "";

/* ------------------------------------------------------------------ sidebar */

function renderNav() {
  const groups = data.categories.map((cat) => {
    const items = data.components.filter((c) => c.category === cat.id && matches(c));
    if (query.trim() && !items.length) return "";
    const isCollapsed = collapsed.has(cat.id);
    return (
      '<div class="group">' +
      '<button class="group-head" data-group="' + esc(cat.id) + '" aria-expanded="' + (!isCollapsed) + '">' +
      '<span class="chev">' + (isCollapsed ? "&#9656;" : "&#9662;") + "</span>" +
      '<span class="cat-icon">' + esc(cat.icon) + "</span>" +
      "<span>" + esc(cat.name) + "</span>" +
      '<span class="count">' + items.length + "</span></button>" +
      (isCollapsed
        ? ""
        : '<div class="group-body">' +
          (items.map((c) => '<button class="nav-item' + (active === c.id ? " active" : "") + '" data-goto="' + esc(c.category + "/" + c.id) + '">' + esc(c.name) + "</button>").join("") ||
            '<p class="muted none">No matches.</p>') +
          "</div>") +
      "</div>"
    );
  }).join("");
  $("#nav").innerHTML = groups || '<p class="muted none">No components match.</p>';
}

/* -------------------------------------------------------------------- views */

function home() {
  active = null;
  crumbs([]);
  const shown = data.components.filter(matches).length;
  $("#content").innerHTML =
    '<section class="hero"><small>PERSONAL RAG TOOLKIT</small>' +
    "<h1>Reusable RAG components.<br>Ready for your projects.</h1>" +
    "<p>Discover canonical Python implementations, inspect contracts and trade-offs, " +
    "and copy code into your own architecture. The registry package is a development-time " +
    "distribution tool, not a client runtime dependency.</p></section>" +
    "<h2>Browse the pipeline</h2>" +
    (query.trim() || filter !== "all"
      ? '<p class="muted">' + shown + " of " + data.components.length + " components match your search and filter.</p>"
      : "") +
    '<div class="grid">' +
    data.categories
      .map((cat) => {
        const n = data.components.filter((c) => c.category === cat.id && matches(c)).length;
        return (
          '<button class="card" data-goto="' + esc(cat.id) + '">' +
          "<strong>" + esc(cat.icon) + " " + esc(cat.name) + "</strong>" +
          '<span class="muted">' + esc(cat.description) + "</span>" +
          '<span class="meta muted">' + n + " component" + (n === 1 ? "" : "s") + "</span></button>"
        );
      })
      .join("") +
    "</div>";
  renderNav();
}

function section(catId) {
  const cat = categoryOf(catId);
  if (!cat) return home();
  active = null;
  crumbs([{ label: cat.name }]);
  const items = data.components.filter((c) => c.category === catId && matches(c));
  $("#content").innerHTML =
    "<h1>" + esc(cat.icon) + " " + esc(cat.name) + "</h1>" +
    '<p class="muted">' + esc(cat.description) + "</p>" +
    '<div class="grid">' +
    (items.map(card).join("") || '<p class="muted">No components match.</p>') +
    "</div>";
  renderNav();
}

let tab = "Overview";

function detail(catId, id) {
  const c = componentOf(id);
  if (!c || c.category !== catId) return section(catId);
  active = id;
  crumbs([{ label: catName(catId), href: catId }, { label: c.name }]);
  let body;
  if (tab === "Code") body = codeTab(c);
  else if (tab === "Dependencies") body = depsTab(c);
  else if (tab === "Trade-offs") body = tradeTab(c);
  else body = overviewTab(c);

  $("#content").innerHTML =
    '<section class="component">' +
    '<div class="muted">' + esc(catName(c.category)) + "</div>" +
    "<h1>" + esc(c.name) + "</h1>" +
    '<p class="lede">' + esc(c.description) + "</p>" +
    '<div class="tags">' + statusTag(c) +
    '<span class="tag">v' + esc(c.version) + "</span>" +
    '<span class="tag">python ' + esc(c.requires_python) + "</span>" +
    (c.dependencies.length ? '<span class="tag">' + c.dependencies.length + " component dep" + (c.dependencies.length === 1 ? "" : "s") + "</span>" : "") +
    "</div>" +
    '<div class="tabs">' +
    ["Overview", "Code", "Dependencies", "Trade-offs"]
      .map((t) => '<button data-tab="' + esc(t) + '" class="' + (t === tab ? "active" : "") + '">' + esc(t) + "</button>")
      .join("") +
    "</div>" + body + "</section>";
  renderNav();
}

const overviewTab = (c) =>
  '<section class="panel"><h2>What it does</h2>' + md(c.purpose || c.description) +
  "</section>" +
  '<section class="panel"><h2>Files</h2><dl class="files">' +
  "<dt>source</dt><dd><code>" + esc(c.source_path) + "</code></dd>" +
  "<dt>tests</dt><dd><code>" + esc(c.test_path) + "</code></dd>" +
  "<dt>documentation</dt><dd><code>" + esc(c.readme_path) + "</code></dd>" +
  "<dt>copies to</dt><dd><code>" + esc(c.destination) + "</code></dd></dl>" +
  '<div class="codebox"><div class="codehead"><strong>Install</strong>' +
  '<button data-copy-install>Copy</button></div><pre><code>' + esc(c.install) + "</code></pre></div></section>";

function codeTab(c) {
  return (
    (c.example
      ? '<section class="panel"><h2>Usage example</h2>' + md(c.example) + "</section>"
      : "") +
    '<div class="codebox"><div class="codehead"><strong>' + esc(c.source_path) + "</strong>" +
    '<button data-copy-code>Copy code</button></div>' +
    '<pre><code>' + esc(c.code) + "</code></pre></div>" +
    '<p class="muted">This is the canonical source. Copying it into your project is the whole point: ' +
    "the component has no runtime dependency on this library.</p>"
  );
}

function depsTab(c) {
  const rows = [];
  rows.push(
    "<dt>Components</dt><dd>" +
      (c.dependencies.length
        ? c.dependencies.map((d) => '<a href="#/' + esc(componentOf(d)?.category || "") + "/" + esc(d) + '" class="pill">' + esc(d) + "</a>").join(" ")
        : '<span class="muted">none — self-contained</span>') +
      "</dd>"
  );
  rows.push(
    "<dt>Required packages</dt><dd>" +
      (c.required_packages.length
        ? c.required_packages.map((p) => '<code class="pill">' + esc(p) + "</code>").join(" ")
        : '<span class="muted">standard library only</span>') +
      "</dd>"
  );
  rows.push(
    "<dt>Optional packages</dt><dd>" +
      (c.optional_packages.length
        ? c.optional_packages.map((p) => '<code class="pill">' + esc(p) + "</code>").join(" ") +
          '<p class="muted">Imported behind a specific ImportError, so the component still loads without them.</p>'
        : '<span class="muted">none</span>') +
      "</dd>"
  );
  rows.push(
    "<dt>External services</dt><dd>" +
      (c.external_services.length
        ? c.external_services
            .map(
              (s) =>
                "<p><strong>" + esc(s.name) + "</strong><br>" + esc(s.notes) +
                '<br><span class="muted">environment: ' +
                s.env_vars.map((v) => "<code>" + esc(v) + "</code>").join(", ") + "</span></p>"
            )
            .join("")
        : '<span class="muted">none — no network access</span>') +
      "</dd>"
  );
  rows.push("<dt>Python</dt><dd>" + esc(c.requires_python) + "</dd>");
  return '<section class="panel"><h2>What you need before using it</h2><dl class="files">' + rows.join("") + "</dl></section>";
}

const tradeTab = (c) =>
  '<section class="panel"><h2>Trade-offs</h2>' + md(c.tradeoffs) + "</section>" +
  '<section class="panel"><h2>Known limitations</h2>' + md(c.limitations) + "</section>";

/* ----------------------------------------------------------------- clipboard */

/* `file://` is a secure context in Chrome and Firefox, so navigator.clipboard
   usually works -- but usually is not a thing to ship. The execCommand fallback
   covers browsers that refuse the async API from a local file, which is
   exactly where this site is meant to be opened. */
async function copyText(text, button) {
  let ok = false;
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      ok = false;
    }
  }
  if (!ok) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:0;left:-9999px";
    document.body.appendChild(ta);
    ta.select();
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    document.body.removeChild(ta);
  }
  const label = button.textContent;
  button.textContent = ok ? "Copied" : "Select and copy manually";
  button.classList.toggle("failed", !ok);
  setTimeout(() => {
    button.textContent = label;
    button.classList.remove("failed");
  }, 1600);
}

/* ------------------------------------------------------------------- events */

document.addEventListener("click", (e) => {
  const goto = e.target.closest("[data-goto]");
  if (goto) {
    tab = "Overview";
    go(goto.dataset.goto);
    return;
  }
  const group = e.target.closest("[data-group]");
  if (group) {
    const id = group.dataset.group;
    collapsed.has(id) ? collapsed.delete(id) : collapsed.add(id);
    renderNav();
    return;
  }
  const t = e.target.closest("[data-tab]");
  if (t) {
    tab = t.dataset.tab;
    route();
    return;
  }
  const copyCode = e.target.closest("[data-copy-code]");
  if (copyCode) {
    copyText(componentOf(active)?.code || "", copyCode);
    return;
  }
  const copyInstall = e.target.closest("[data-copy-install]");
  if (copyInstall) {
    copyText(componentOf(active)?.install || "", copyInstall);
  }
});

$("#search").addEventListener("input", (e) => {
  query = e.target.value;
  route();
});
$("#filter").addEventListener("change", (e) => {
  filter = e.target.value;
  route();
});
window.addEventListener("hashchange", route);

/* Populate the filter from the same data as everything else. */
$("#filter").innerHTML = FILTERS.map(
  ([value, label]) => '<option value="' + value + '">' + esc(label) + "</option>"
).join("");

route();
})();
