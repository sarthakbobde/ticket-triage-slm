"""Step 04a: build the hand-labelling sheet for the 200 gold tickets.

Output: data/gold/label_gold.html  (one file, open it in your browser; no server needed)

The sheet shows ONLY the ticket text. It never shows the dataset's labels or the teacher's labels,
so nothing can anchor you. You choose category, priority, sentiment, and type any product names.
Your answers are saved in the browser as you click, and "Export" downloads them as a file.

Run:  uv run python scripts/04_make_gold_sheet.py
"""

import json
import random
import sys
from pathlib import Path

from triage.prompts import CATEGORY_DEFINITIONS
from triage.schema import Category, Sentiment

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "processed" / "test.jsonl"
OUT_FILE = ROOT / "data" / "gold" / "label_gold.html"

# Same rubric as SYSTEM_PROMPT in src/triage/prompts.py (keep the two in sync).
PRIORITY_RULES = {
    "high": "customer blocked, no workaround",
    "medium": "problem with a workaround, or time-sensitive question",
    "low": "general question, feedback, feature request",
}

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold set labelling</title>
<style>
  :root { --bg:#f7f7f5; --card:#fff; --text:#1c1c1a; --muted:#6b6b66; --line:#dcdcd6;
          --accent:#2457d6; --accent-text:#fff; --done:#2e7d32; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --card:#222221; --text:#ececea; --muted:#9a9a94; --line:#3a3a37;
            --accent:#6b93ff; --accent-text:#101010; --done:#66bb6a; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 16px; }
  header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; }
  h1 { font-size: 18px; margin: 0; }
  .bar { height:6px; background:var(--line); border-radius:3px; margin:8px 0 16px; overflow:hidden; }
  .bar div { height:100%; background:var(--done); width:0; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:12px; }
  #ticket { white-space:pre-wrap; word-wrap:break-word; font-size:16px; max-height:42vh; overflow:auto; }
  .row { margin: 10px 0; }
  .label { font-size:13px; color:var(--muted); margin-bottom:4px; }
  .choices { display:flex; flex-wrap:wrap; gap:8px; }
  button { font:inherit; color:var(--text); background:var(--card); border:1px solid var(--line);
           border-radius:8px; padding:8px 12px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.on { background:var(--accent); color:var(--accent-text); border-color:var(--accent); }
  kbd { font-size:12px; opacity:.7; margin-right:4px; }
  input[type=text] { width:100%; font:inherit; padding:8px 10px; border-radius:8px;
                     border:1px solid var(--line); background:var(--card); color:var(--text); }
  .nav { display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; }
  details { font-size:14px; } summary { cursor:pointer; color:var(--muted); }
  details li { margin:4px 0; }
  .muted { color:var(--muted); font-size:13px; }
  #status { color:var(--done); }
</style>
</head>
<body>
<main>
  <header>
    <h1>Gold set: <span id="pos"></span></h1>
    <div class="muted"><span id="count"></span> labelled <span id="status"></span></div>
  </header>
  <div class="bar"><div id="fill"></div></div>

  <div class="card"><div id="ticket"></div></div>

  <div class="card">
    <div class="row"><div class="label">Category (keys 1-5): is something broken? then technical.</div>
      <div class="choices" id="category"></div></div>
    <div class="row"><div class="label">Priority (keys L / M / H)</div>
      <div class="choices" id="priority"></div></div>
    <div class="row"><div class="label">Sentiment (keys N negative / U neutral / P positive)</div>
      <div class="choices" id="sentiment"></div></div>
    <div class="row"><div class="label">Products named in the ticket, comma separated (leave empty if none)</div>
      <input type="text" id="products" placeholder="e.g. Jira, MacBook Pro"></div>
  </div>

  <div class="nav">
    <div class="choices">
      <button id="prev"><kbd>&larr;</kbd>Back</button>
      <button id="next">Next<kbd>&rarr;</kbd></button>
      <button id="nextEmpty">Next unlabelled</button>
    </div>
    <div class="choices">
      <button id="export">Export file</button>
      <button id="import">Import file</button>
      <input type="file" id="file" accept=".jsonl,.json" hidden>
    </div>
  </div>

  <details class="card" style="margin-top:12px">
    <summary>Guidelines (the same rules the teacher used)</summary>
    <p><b>Category</b></p><ul id="catRules"></ul>
    <p><b>Priority</b></p><ul id="priRules"></ul>
    <p class="muted">Label from the ticket text alone. Do not guess what the dataset or the teacher said.</p>
  </details>
  <p class="muted">Answers are saved in this browser automatically. Click "Export file" when you stop,
    and again when you finish. Keep the same browser to carry on later, or use "Import file".</p>
</main>

<script>
const TICKETS = __TICKETS__;
const CATEGORIES = __CATEGORIES__;
const PRIORITIES = __PRIORITIES__;
const SENTIMENTS = __SENTIMENTS__;
const CAT_RULES = __CAT_RULES__;
const PRI_RULES = __PRI_RULES__;
const STORE_KEY = "gold_labels_v1";

let answers = {};   // ticket id -> {category, priority, sentiment, products}
let index = 0;

function load() {
  try { answers = JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch (e) { answers = {}; }
}
function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(answers)); } catch (e) {}
}
function isDone(a) { return a && a.category && a.priority && a.sentiment; }
function current() { return TICKETS[index]; }
function answerFor(id) { if (!answers[id]) answers[id] = { products: "" }; return answers[id]; }

function makeButtons(containerId, values, field, keys) {
  const box = document.getElementById(containerId);
  box.innerHTML = "";
  values.forEach((value, i) => {
    const b = document.createElement("button");
    b.dataset.value = value;
    b.innerHTML = "<kbd>" + keys[i] + "</kbd>" + value.replace("_", " ");
    b.onclick = () => choose(field, value);
    box.appendChild(b);
  });
}
function choose(field, value) {
  answerFor(current().id)[field] = value;
  save(); render();
}
function render() {
  const t = current();
  const a = answers[t.id] || {};
  document.getElementById("pos").textContent = (index + 1) + " / " + TICKETS.length;
  document.getElementById("ticket").textContent = t.text;
  document.getElementById("ticket").scrollTop = 0;
  for (const field of ["category", "priority", "sentiment"]) {
    document.querySelectorAll("#" + field + " button").forEach(b => {
      b.classList.toggle("on", a[field] === b.dataset.value);
    });
  }
  document.getElementById("products").value = a.products || "";
  let done = 0;
  TICKETS.forEach(x => { if (isDone(answers[x.id])) done++; });
  document.getElementById("count").textContent = done + " of " + TICKETS.length;
  document.getElementById("fill").style.width = (100 * done / TICKETS.length) + "%";
}
function go(step) {
  index = Math.min(TICKETS.length - 1, Math.max(0, index + step));
  render();
}
function goNextEmpty() {
  for (let k = 1; k <= TICKETS.length; k++) {
    const i = (index + k) % TICKETS.length;
    if (!isDone(answers[TICKETS[i].id])) { index = i; render(); return; }
  }
  flash("All labelled. Export the file.");
}
function flash(message) {
  const s = document.getElementById("status");
  s.textContent = " - " + message;
  setTimeout(() => { s.textContent = ""; }, 3000);
}
function exportFile() {
  const lines = [];
  TICKETS.forEach(t => {
    const a = answers[t.id];
    if (!isDone(a)) return;
    const products = (a.products || "").split(",").map(p => p.trim()).filter(p => p);
    lines.push(JSON.stringify({ id: t.id, category: a.category, priority: a.priority,
                                sentiment: a.sentiment, products: products }));
  });
  const blob = new Blob([lines.join("\n") + "\n"], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "gold_labels.jsonl";
  link.click();
  flash(lines.length + " labels exported");
}
function importFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let n = 0;
    reader.result.split("\n").forEach(line => {
      if (!line.trim()) return;
      const row = JSON.parse(line);
      answers[row.id] = { category: row.category, priority: row.priority,
                          sentiment: row.sentiment, products: (row.products || []).join(", ") };
      n++;
    });
    save(); render(); flash(n + " labels imported");
  };
  reader.readAsText(file);
}

load();
makeButtons("category", CATEGORIES, "category", ["1", "2", "3", "4", "5"]);
makeButtons("priority", PRIORITIES, "priority", ["L", "M", "H"]);
makeButtons("sentiment", SENTIMENTS, "sentiment", ["N", "U", "P"]);
CAT_RULES.forEach(r => { const li = document.createElement("li");
  li.innerHTML = "<b>" + r[0] + "</b>: " + r[1]; document.getElementById("catRules").appendChild(li); });
PRI_RULES.forEach(r => { const li = document.createElement("li");
  li.innerHTML = "<b>" + r[0] + "</b>: " + r[1]; document.getElementById("priRules").appendChild(li); });

document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("nextEmpty").onclick = goNextEmpty;
document.getElementById("export").onclick = exportFile;
document.getElementById("import").onclick = () => document.getElementById("file").click();
document.getElementById("file").onchange = e => { if (e.target.files[0]) importFile(e.target.files[0]); };
document.getElementById("products").oninput = e => {
  answerFor(current().id).products = e.target.value; save();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "arrowright" || key === "enter") go(1);
  else if (key === "arrowleft") go(-1);
  else if (key >= "1" && key <= String(CATEGORIES.length)) choose("category", CATEGORIES[Number(key) - 1]);
  else if (key === "l") choose("priority", "low");
  else if (key === "m") choose("priority", "medium");
  else if (key === "h") choose("priority", "high");
  else if (key === "n") choose("sentiment", "negative");
  else if (key === "u") choose("sentiment", "neutral");
  else if (key === "p") choose("sentiment", "positive");
});
// Start at the first ticket that is not labelled yet.
for (let i = 0; i < TICKETS.length; i++) { if (!isDone(answers[TICKETS[i].id])) { index = i; break; } }
render();
</script>
</body>
</html>
"""


def main():
    # 1. The gold tickets, text only. No dataset labels, no teacher labels.
    gold = []
    with open(TEST_FILE, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["gold"]:
                gold.append({"id": row["id"], "text": row["text"]})

    # 2. Mix the order (the file order may follow the dataset's queues, which could hint at labels).
    random.seed(SEED)
    random.shuffle(gold)

    # 3. Fill the page template.
    categories = []
    category_rules = []
    for category in Category:
        categories.append(category.value)
        category_rules.append([category.value, CATEGORY_DEFINITIONS[category]])
    priorities = ["low", "medium", "high"]
    priority_rules = []
    for name in ["high", "medium", "low"]:
        priority_rules.append([name, PRIORITY_RULES[name]])
    sentiments = []
    for sentiment in Sentiment:
        sentiments.append(sentiment.value)

    page = PAGE
    page = page.replace("__TICKETS__", json.dumps(gold, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__CATEGORIES__", json.dumps(categories))
    page = page.replace("__PRIORITIES__", json.dumps(priorities))
    page = page.replace("__SENTIMENTS__", json.dumps(sentiments))
    page = page.replace("__CAT_RULES__", json.dumps(category_rules))
    page = page.replace("__PRI_RULES__", json.dumps(priority_rules))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(page, encoding="utf-8")
    print(f"{len(gold)} gold tickets -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
