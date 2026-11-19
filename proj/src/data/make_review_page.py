"""生成术语表人工裁决页面（单文件 HTML，离线可用，数据内嵌）。

产出：reports/glossary_review.html
  - EMEA 页：400 条候选，带 2 条语料实证句 + test 影响句数，三分类裁决（保留/改译/剔除）
  - MTNT 页：200 条俚语候选，带 3 条带金标译文的训练对，填译法 + 处理方式
  - 进度存 localStorage，导出 reviewed.json 后交给 build_glossary.py --mode finalize

用法：
  python src/data/make_review_page.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "glossary_review.html"


def load(name: str) -> list[dict]:
    path = DATA / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def attach_emea(cands: list[dict], train: list[dict], dev: list[dict], test: list[dict]) -> list[dict]:
    """给每条候选补 2 条语料实证 + dev/test 影响句数。"""
    idx: dict[str, list[int]] = {}
    for i, r in enumerate(train):
        idx.setdefault(r["src"].lower(), []).append(i)
    test_src = [r["src"].lower() for r in test]
    dev_src = [r["src"].lower() for r in dev]

    out = []
    for c in cands:
        en = c["en"].lower()
        examples = []
        seen = set()
        for i, r in enumerate(train):
            if en in r["src"].lower() and r["tgt"][:80] not in seen:
                seen.add(r["tgt"][:80])
                examples.append({"en": r["src"][:220], "fr": r["tgt"][:220]})
                if len(examples) == 2:
                    break
        hits = sum(1 for s in test_src if en in s)
        dev_hits = sum(1 for s in dev_src if en in s)
        en_words = len(c["en"].split())
        fr_words = len(c["fr"].split())
        out.append({**{k: c[k] for k in ("en", "fr", "score", "cooc", "en_freq", "fr_freq")},
                    "n_tokens": (c.get("variants") or [{}])[0].get("n_tokens", 0),
                    "single_token": (c.get("variants") or [{}])[0].get("single_token", False),
                    "en_words": en_words, "fr_words": fr_words,
                    "mismatch": abs(en_words - fr_words) >= 1,
                    "test_hits": hits, "dev_hits": dev_hits,
                    "examples": examples})
    return out


def attach_slang(cands: list[dict], train: list[dict], test: list[dict]) -> list[dict]:
    out = []
    for c in cands:
        tok_ = c["token"]
        pat = re.compile(r"(^|\W)" + re.escape(tok_) + r"(\W|$)", re.I)
        examples = []
        for r in train:
            if pat.search(r["src"]):
                examples.append({"en": r["src"][:220], "fr": r["tgt"][:220]})
                if len(examples) == 3:
                    break
        hits = sum(1 for r in test if pat.search(r["src"]))
        out.append({**c, "test_hits": hits, "examples": examples})
    return out


payload = {
    "meta": {},
    "emea": [],
    "mtnt": [],
}

emea = load("glossary.emea.candidates.jsonl")
slang = load("glossary.mtnt.candidates.jsonl")
emea_train = load("emea.train.jsonl")
emea_dev = load("emea.dev.jsonl")
emea_test = load("emea.test.jsonl")
mtnt_train = load("mtnt.train.jsonl")
mtnt_test = load("mtnt.test.jsonl")

if emea:
    payload["emea"] = attach_emea(emea, emea_train, emea_dev, emea_test)
if slang:
    payload["mtnt"] = attach_slang(slang, mtnt_train, mtnt_test)

payload["meta"] = {
    "emea_total": len(payload["emea"]),
    "mtnt_total": len(payload["mtnt"]),
    "emea_test_rows": len(emea_test),
    "mtnt_test_rows": len(mtnt_test),
    "generated_from": ["glossary.emea.candidates.jsonl", "glossary.mtnt.candidates.jsonl"],
}

JSON_BLOB = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>术语表人工裁决 — CS5489 En-Fr</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#fff; --ink:#1f2933; --muted:#6b7785; --line:#e3e7ec;
    --keep:#0f766e; --fix:#b45309; --drop:#b91c1c; --accent:#1d4ed8; --soft:#eef2ff;
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",Roboto,sans-serif;
       background:var(--bg);color:var(--ink)}
  header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line);
         padding:12px 18px;box-shadow:0 2px 8px rgba(16,24,40,.04)}
  h1{font-size:16px;margin:0 0 6px}
  .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .chip{background:var(--soft);color:var(--accent);border-radius:999px;padding:2px 10px;font-size:12px}
  .chip.g{background:#ecfdf5;color:var(--keep)} .chip.y{background:#fffbeb;color:var(--fix)}
  .chip.r{background:#fef2f2;color:var(--drop)}
  .progress{flex:1;min-width:180px;height:8px;background:#eef1f5;border-radius:999px;overflow:hidden}
  .progress > i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
  button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;
         padding:6px 12px;cursor:pointer}
  button:hover{background:#f2f5f9}
  button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  button.keep{border-color:var(--keep);color:var(--keep)}
  button.keep.on{background:var(--keep);color:#fff}
  button.fix{border-color:var(--fix);color:var(--fix)}
  button.fix.on{background:var(--fix);color:#fff}
  button.drop{border-color:var(--drop);color:var(--drop)}
  button.drop.on{background:var(--drop);color:#fff}
  input[type=text],select{font:inherit;padding:6px 9px;border:1px solid var(--line);
                          border-radius:8px;background:#fff;width:100%}
  main{padding:16px 18px 80px;max-width:1180px;margin:0 auto}
  .tabs{display:flex;gap:8px;margin-bottom:12px}
  .tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
  .toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
  .toolbar .grow{flex:1;min-width:200px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:12px 14px;margin-bottom:10px}
  .card.done-true{border-left:4px solid var(--keep)}
  .card.done-fix{border-left:4px solid var(--fix)}
  .card.done-false{border-left:4px solid var(--drop)}
  .row{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}
  .pair{flex:1;min-width:260px}
  .en{font-weight:600}
  .fr{color:#0b4f6c}
  .kv{color:var(--muted);font-size:12px;display:flex;gap:10px;flex-wrap:wrap;margin-top:4px}
  .badge{background:#fef3c7;color:#92400e;border-radius:6px;padding:1px 7px;font-size:11.5px}
  .badge.warn{background:#fee2e2;color:#b91c1c}
  .warn{color:#b45309}
  details{margin-top:8px}
  summary{cursor:pointer;color:var(--muted);font-size:12px}
  .ex{background:#fafbfc;border:1px dashed var(--line);border-radius:8px;
      padding:6px 9px;margin-top:6px;font-size:12.5px;color:#374151}
  .ex b{color:var(--muted);font-weight:500}
  .acts{display:flex;gap:6px;align-items:center}
  .hint{color:var(--muted);font-size:12px;margin-top:10px}
  kbd{background:#f1f3f6;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
      padding:0 4px;font-size:11.5px}
  .hidden{display:none}
  .empty{text-align:center;color:var(--muted);padding:60px 0}
  footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid var(--line);
         padding:10px 18px;display:flex;gap:10px;align-items:center;justify-content:space-between}
  @media (max-width:720px){.acts{width:100%}.pair{min-width:100%}}
</style>
</head>
<body>
<header>
  <h1>术语表人工裁决 <span class="chip" id="hdr"></span></h1>
  <div class="bar">
    <span class="chip g" id="c-keep">保留 0</span>
    <span class="chip y" id="c-fix">改译 0</span>
    <span class="chip r" id="c-drop">剔除 0</span>
    <span class="chip" id="c-left">未判 0</span>
    <div class="progress"><i id="pg"></i></div>
    <button id="btn-export" class="primary">导出 reviewed.json</button>
  </div>
</header>

<main>
  <div class="tabs">
    <button data-tab="emea" class="on" id="tab-emea">EMEA 术语</button>
    <button data-tab="mtnt" id="tab-mtnt">MTNT 俚语</button>
  </div>

  <div class="toolbar">
    <span class="grow"><input type="text" id="q" placeholder="搜索英文/法文/例句…"></span>
    <select id="filter" style="width:auto">
      <option value="all">全部</option>
      <option value="todo">只看未判</option>
      <option value="1">只看保留</option>
      <option value="2">只看改译</option>
      <option value="0">只看剔除</option>
    </select>
    <button id="btn-sort">按影响排序</button>
    <button id="btn-clear">清空本页进度</button>
  </div>

  <div id="list"></div>
</main>

<footer>
  <span class="hint">
    <kbd>1</kbd> 保留　<kbd>2</kbd> 改译（输入框自动聚焦）　<kbd>0</kbd> 剔除　
    <kbd>j</kbd>/<kbd>k</kbd> 上下条　进度自动保存在本机浏览器
  </span>
  <span class="chip" id="f-hint">EMEA：先看例句再判断；MTNT：按训练对里的实际译法填写</span>
</footer>

<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const KEY = 'cs5489-glossary-review-v1';
let state = JSON.parse(localStorage.getItem(KEY) || '{"emea":{},"mtnt":{}}');
let tab = 'emea', sortByImpact = false;

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }

function cur(){ return tab === 'emea' ? DATA.emea : DATA.mtnt; }

function rec(id){
  state[tab] = state[tab] || {};
  return state[tab][id] || (state[tab][id] = {decision:null, fr:'', mode:''});
}

function stats(){
  const items = cur();
  const s = {keep:0, fix:0, drop:0, left:0};
  items.forEach(it => {
    const d = rec(it.id || it.en || it.token).decision;
    if (d === 1) s.keep++; else if (d === 2) s.fix++; else if (d === 0) s.drop++; else s.left++;
  });
  return s;
}

function refreshStats(){
  const s = stats();
  document.getElementById('c-keep').textContent = '保留 ' + s.keep;
  document.getElementById('c-fix').textContent = '改译 ' + s.fix;
  document.getElementById('c-drop').textContent = '剔除 ' + s.drop;
  document.getElementById('c-left').textContent = '未判 ' + s.left;
  const total = cur().length || 1;
  document.getElementById('pg').style.width = (100 * (total - s.left) / total) + '%';
  document.getElementById('hdr').textContent =
    tab === 'emea'
      ? (DATA.meta.emea_total + ' 条候选 / test 影响 ' + DATA.meta.emea_test_rows + ' 句')
      : (DATA.meta.mtnt_total + ' 条候选 / test ' + DATA.meta.mtnt_test_rows + ' 句');
}

function esc(s){ return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function keyOf(it){ return it.id || it.en || it.token; }

function visible(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const f = document.getElementById('filter').value;
  let items = cur().filter(it => {
    const d = rec(keyOf(it)).decision;
    if (f === 'todo' && d !== null && d !== undefined) return false;
    if (f === '1' && d !== 1) return false;
    if (f === '2' && d !== 2) return false;
    if (f === '0' && d !== 0) return false;
    if (!q) return true;
    const hay = [it.en, it.fr, it.token, it.fr0, it.notes,
                 (it.examples||[]).map(e => e.en + ' ' + e.fr).join(' ')]
                 .filter(Boolean).join(' ').toLowerCase();
    return hay.includes(q);
  });
  if (sortByImpact) items = items.slice().sort((a,b) => (b.test_hits||0) - (a.test_hits||0));
  return items;
}

function render(){
  refreshStats();
  const items = visible();
  const list = document.getElementById('list');
  if (!items.length){ list.innerHTML = '<div class="empty">没有符合条件的条目</div>'; return; }
  list.innerHTML = items.map(it => tab === 'emea' ? cardEmea(it) : cardMtnt(it)).join('');
  bind();
}

function cardEmea(it){
  const k = it.en, r = rec(k);
  const cls = r.decision === 1 ? 'done-true' : r.decision === 2 ? 'done-fix' : r.decision === 0 ? 'done-false' : '';
  const ex = (it.examples||[]).map(e =>
      `<div class="ex"><b>EN</b> ${esc(e.en)}<br><b>FR</b> ${esc(e.fr)}</div>`).join('');
  return `<div class="card ${cls}" data-key="${esc(k)}">
    <div class="row">
      <div class="pair">
        <div class="en">${esc(it.en)}</div>
        <div class="fr" style="margin-top:4px">
          <input type="text" class="frin" value="${esc(r.fr || it.fr)}" data-key="${esc(k)}">
        </div>
        <div class="kv">
          <span>score ${it.score}</span><span>共现 ${it.cooc}</span>
          <span>${it.en_words}→${it.fr_words} 词${it.single_token ? ' · 单token' : ''}</span>
          <span class="${it.test_hits ? '' : 'warn'}">dev ${it.dev_hits} / test ${it.test_hits} 句</span>
          ${it.mismatch ? '<span class="badge">词数不一致</span>' : ''}
          ${(it.test_hits + it.dev_hits) === 0 ? '<span class="badge warn">语料不出现</span>' : ''}
        </div>
        ${ex ? `<details><summary>语料实证（2 条）</summary>${ex}</details>` : ''}
      </div>
      <div class="acts">
        <button class="keep ${r.decision===1?'on':''}" data-act="1" data-key="${esc(k)}">保留</button>
        <button class="fix ${r.decision===2?'on':''}" data-act="2" data-key="${esc(k)}">改译</button>
        <button class="drop ${r.decision===0?'on':''}" data-act="0" data-key="${esc(k)}">剔除</button>
      </div>
    </div>
  </div>`;
}

function cardMtnt(it){
  const k = it.token, r = rec(k);
  const cls = r.decision === 1 ? 'done-true' : r.decision === 2 ? 'done-fix' : r.decision === 0 ? 'done-false' : '';
  const ex = (it.examples||[]).map(e =>
      `<div class="ex"><b>EN</b> ${esc(e.en)}<br><b>FR</b> ${esc(e.fr)}</div>`).join('');
  const modes = [['keep', '保留原样'], ['translate', '意译'], ['omit', '省略/改写']];
  return `<div class="card ${cls}" data-key="${esc(k)}">
    <div class="row">
      <div class="pair">
        <div class="en">${esc(it.token)}</div>
        <div class="kv">
          <span>MTNT ${it.mtnt_freq}</span><span>EMEA ${it.emea_freq}</span>
          <span>clean ${it.clean_freq}</span><span>test 命中 ${it.test_hits} 句</span>
        </div>
        <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
          <span style="flex:1;min-width:200px">
            <input type="text" class="frin" placeholder="法语译法（按训练对里的实际写法填）"
                   value="${esc(r.fr)}" data-key="${esc(k)}">
          </span>
          <span style="width:150px">
            <select class="modein" data-key="${esc(k)}">
              ${modes.map(([v,t]) =>
                `<option value="${v}" ${r.mode===v?'selected':''}>${t}</option>`).join('')}
            </select>
          </span>
        </div>
        ${ex ? `<details open><summary>带金标译文的训练对（最多 3 条）</summary>${ex}</details>` : ''}
      </div>
      <div class="acts">
        <button class="keep ${r.decision===1?'on':''}" data-act="1" data-key="${esc(k)}">收下</button>
        <button class="drop ${r.decision===0?'on':''}" data-act="0" data-key="${esc(k)}">剔除</button>
      </div>
    </div>
  </div>`;
}

function bind(){
  document.querySelectorAll('button[data-act]').forEach(b => {
    b.onclick = () => {
      const k = b.dataset.key, r = rec(k);
      r.decision = Number(b.dataset.act);
      if (r.decision === 2){
        const inp = document.querySelector(`input.frin[data-key="${CSS.escape(k)}"]`);
        if (inp) inp.focus();
        if (!r.fr){
          const card = b.closest('.card');
          const cur0 = card.querySelector('input.frin').value || '';
          r.fr = cur0;
        }
      }
      save(); render();
    };
  });
  document.querySelectorAll('input.frin').forEach(inp => {
    inp.oninput = () => { const r = rec(inp.dataset.key); r.fr = inp.value; save(); };
    inp.onkeydown = e => { if (e.key === 'Enter') inp.blur(); };
  });
  document.querySelectorAll('select.modein').forEach(sel => {
    sel.onchange = () => { const r = rec(sel.dataset.key); r.mode = sel.value; save(); };
  });
}

/* 键盘：在全页生效，但输入框内不拦截 */
document.addEventListener('keydown', e => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
  if (!['0','1','2','j','k'].includes(e.key)) return;
  const items = visible();
  if (!items.length) return;
  const cards = [...document.querySelectorAll('.card')];
  let idx = cards.findIndex(c => c.getBoundingClientRect().top > 68);
  if (idx < 0) idx = cards.length - 1;
  const card = e.key === 'k' ? cards[Math.max(0, idx - 1)] : cards[idx];
  if (!card) return;
  if (e.key === 'j'){ card.scrollIntoView({block:'start', behavior:'smooth'}); return; }
  if (e.key === 'k'){ card.scrollIntoView({block:'start', behavior:'smooth'}); return; }
  const k = card.dataset.key, r = rec(k);
  r.decision = Number(e.key);
  if (r.decision === 2){
    const inp = card.querySelector('input.frin');
    if (!r.fr && inp) r.fr = inp.value;
    if (inp) inp.focus();
  }
  save(); render();
});

document.querySelectorAll('.tabs button').forEach(b => {
  b.onclick = () => {
    tab = b.dataset.tab;
    document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
    document.getElementById('f-hint').textContent = tab === 'emea'
      ? 'EMEA：先看例句再判断；改译时把规范译法写进输入框'
      : 'MTNT：照训练对里的实际译法填写，别自己造';
    render();
  };
});
document.getElementById('q').oninput = render;
document.getElementById('filter').onchange = render;
document.getElementById('btn-sort').onclick = () => { sortByImpact = !sortByImpact; render(); };
document.getElementById('btn-clear').onclick = () => {
  if (!confirm('清空当前标签页的裁决记录？')) return;
  state[tab] = {}; save(); render();
};
document.getElementById('btn-export').onclick = () => {
  const out = {meta: {...DATA.meta, exported_at: new Date().toISOString()}, emea: [], mtnt: []};
  DATA.emea.forEach(it => {
    const r = rec2('emea', it.en);
    if (!r) return;
    out.emea.push({en: it.en, fr_original: it.fr, fr_final: (r.fr || it.fr).trim(),
                   decision: r.decision, score: it.score, cooc: it.cooc,
                   test_hits: it.test_hits, single_token: it.single_token});
  });
  DATA.mtnt.forEach(it => {
    const r = rec2('mtnt', it.token);
    if (!r) return;
    out.mtnt.push({token: it.token, fr: (r.fr || '').trim(), decision: r.decision,
                   mode: r.mode || 'translate', mtnt_freq: it.mtnt_freq, test_hits: it.test_hits});
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'reviewed.json';
  a.click();
  URL.revokeObjectURL(a.href);
};
function rec2(which, k){
  const m = state[which] || {};
  const r = m[k];
  return (r && r.decision !== null && r.decision !== undefined) ? r : null;
}

/* 给每条补一个稳定的 key 字段，供 rec() 使用 */
DATA.emea.forEach(it => { it.id = it.en; });
DATA.mtnt.forEach(it => { it.id = it.token; });

render();
</script>
</body>
</html>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
html = TEMPLATE.replace("__PAYLOAD__", JSON_BLOB)
OUT.write_text(html, encoding="utf-8")
print(f"[ok] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"     EMEA {len(payload['emea'])} 条，MTNT {len(payload['mtnt'])} 条")
if payload["emea"]:
    top = sorted(payload["emea"], key=lambda d: -d["test_hits"])[:5]
    print("     test 影响最大的 5 条：" + "，".join(f"{d['en']}({d['test_hits']})" for d in top))

# 自检：把生成的文件读回来，确认内嵌 JSON 可解析、字段齐全
m = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S)
back = json.loads(m.group(1))
assert len(back["emea"]) == len(payload["emea"]), "EMEA 条数不一致"
assert len(back["mtnt"]) == len(payload["mtnt"]), "MTNT 条数不一致"
for it in back["emea"][:5]:
    assert {"en", "fr", "score", "test_hits", "n_tokens", "examples"} <= set(it), it.keys()
missing_ex = sum(1 for x in back["emea"] if not x["examples"])
print(f"[check] 内嵌 JSON 可解析；无例句的 EMEA 条 {missing_ex} 条，"
      f"无例句的 MTNT 条 {sum(1 for x in back['mtnt'] if not x['examples'])} 条")

