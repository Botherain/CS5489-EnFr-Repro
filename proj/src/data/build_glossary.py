"""S1.3 术语表构建：EMEA 统计对齐 + MTNT 俚语跨域对比。

对齐以"内容词签名"（排序去重后的词集合）为主键，单位与数字在两侧都不计入内容词，
因此 `body weight/day` 不会被拆成 `body weight day`，`tablets Sitagliptin` 与
`Sitagliptin tablets` 也不会重复入表。每个英文签名只保留一个法语签名，反之亦然。

输出附带 signature，命中率匹配可做词序无关比较。

用法：
  python src/data/build_glossary.py --mode align --top 600
  python src/data/build_glossary.py --mode slang --top 200
  python src/data/build_glossary.py --mode audit
  python src/data/build_glossary.py --mode llm --samples 120
  python src/data/build_glossary.py --mode coverage
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = DATA

EN_TOK = re.compile(r"[A-Za-z][A-Za-z\-']*")
FR_TOK = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ][A-Za-zÀ-ÖØ-öø-ÿŒœÆæ\-']*")

STOP = set("""a an the of in on at to for with and or by from as is are was were be been being
this that these those it its their his her our your my we you they he she i not no than then
which who whom whose what when where why how all any both each few more most other some such
only own same so too very can will just should now during after before above below between
into through against under over out up down off again further once here there also may might
must shall would could per via e g ie eg vs etc page emea""".split())

# EMEA 里高频出现但不构成术语的样板词
BOILER = set("""european medicines agency human use product information annex summary
characteristics marketing authorisation holder medicinal product refer to section see
should be not known frequency very common uncommon rare adverse reaction patients treatment""".split())

# PDF 页眉页脚碎片：出现在术语里一定是抽取噪声
PDF_NOISE = {"article", "total", "leaflet", "page", "pages", "date", "version",
             "printed", "contents", "footer", "header", "updated", "last", "renewed",
             "contains", "each", "chaque", "contient", "objectives",
             "january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december"}

# 说明书指令性套话：整条丢弃
BOILER_VERB = {"take", "use", "used", "store", "keep", "tell", "ask", "stop", "call", "read",
               "check", "wait", "try", "give", "put", "mix", "swallow", "repeat", "see",
               "select", "choose", "remove", "open", "close", "hold", "insert", "attach",
               "prepare", "administer", "shake", "wash", "wipe", "discard", "allow", "warm",
               "renew", "renewed", "inject", "apply", "replace", "dispose", "return", "contact"}
FR_VERB = {"utilisé", "utiliser", "choisissez", "sélectionnez", "conservez", "prenez", "vérifiez",
           "jetez", "retirez", "appuyez", "tournez", "administrez", "préparez", "rincez",
           "nettoyez", "ouvrez", "fermez", "contactez", "renouvelée", "renouvelé"}
BOILER_PREFIX = ("your ", "the doctor", "do not", "if you", "you should", "you must",
                 "tell your", "ask your", "before you", "after you", "this medicine",
                 "this product", "the medicine", "the product", "the patient", "it is used")

# 单位与剂量词（两侧都不算内容词）
UNIT = set("""mg mcg µg ug ng g kg ml cl dl l iu ui miu mmol mol µm um nm cm mm m
% cfu kcal kj da kda ppm meq eq bq mbq gb kb mb""".split())

NUM_RE = re.compile(r"^[\d.,/:\-–—+]+$")
UNIT_RE = re.compile(r"^(" + "|".join(sorted(map(re.escape, UNIT), key=len, reverse=True)) +
                     r")(/(ml|kg|g|l|m2|cm2|dose|day|h|min))?$", re.I)

FR_STOP = set("""de du des la le les l d un une et ou pour dans sur avec par au aux en qu que qui
quoi dont est sont être ont cette ce ces son sa ses leur leurs il elle ils elles on nous vous je tu
plus moins très tout tous toute toutes même aussi comme lors chez sans sous entre selon après avant
pendant depuis vers contre y ne pas se c n m t j s""".split())

EDGE = set("""of the and a an to for with in on by from was were is are be been that this
these those as at or nor but if then than which who whom whose what when where why how""".split())

# 法语省音字母（被 tokenizer 切开后需要粘回）
ELISION = re.compile(r"\b(d|l|n|m|t|s|c|j|qu)\s+([a-zà-ÿ])")
POSSESS = re.compile(r"\b([A-Z][a-z]{2,})\s+s\s+(?=[a-z])")


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def normalize_text(text: str, side: str) -> str:
    """在分词前修正 PDF 抽取造成的撇号断裂。

    法语：`d ' injection` / `d 'élimination` -> `d'injection`
    英语：`Kaposi ' s` / `Kaposi s sarcoma` -> `Kaposi's`
    按语言分别处理：法语 `s'` 合法（s'il），英语 `X s Y` 是领属格。
    """
    s = text.replace("\u2019", "'").replace("\u00a0", " ")
    if side == "fr":
        s = re.sub(r"\b([dlmntcj])\s*'\s*", r"\1'", s)
    else:
        s = re.sub(r"([A-Za-z])\s+'\s*s\b", r"\1's", s)
        s = re.sub(r"\b([A-Z][a-z]{2,})\s+s\s+(?=[a-z])", r"\1's ", s)
    return re.sub(r"[ \t]+", " ", s)


def repair(surface: str) -> str:
    """展示用：只做空白与首尾清理，不再改撇号（改在 normalize_text 里做）。"""
    return re.sub(r"\s+", " ", surface).strip()


def clean_surface(surface: str) -> str:
    """展示串里去掉单位与裸数字：'Advagraf mg gélules' -> 'Advagraf gélules'。"""
    toks = [t for t in surface.split()
            if not UNIT_RE.match(t.strip(".,;:")) and not NUM_RE.match(t.strip(".,;:"))]
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


BRAND_RE = re.compile(r"^[A-Z]{5,}$")


def looks_brand(en: str) -> bool:
    """商标/公司名：全大写长串，或每个词都首字母大写且没有小写词（Nobilis Influenza H）。"""
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z\-']*", en) if not UNIT_RE.match(t)]
    if not toks:
        return True
    if any(BRAND_RE.match(t) for t in toks):                 # ADARTREL / HEXAL / SIFROL
        return True
    if len(toks) == 1:
        t = toks[0]
        if t.isupper():                                      # B-CLL / LLC 这类缩写保留
            return False
        return t[:1].isupper() and len(t) > 3
    return all(t[:1].isupper() for t in toks) and not any(t.islower() for t in toks)


MED_SUFFIX = ("itis", "osis", "aemia", "emia", "pathy", "plasty", "ectomy", "otomy", "ase",
              "ium", "one", "oma", "ics", "ism", "mab", "nib", "vir", "mycin", "cillin",
              "pril", "sartan", "statin", "azole", "tidine", "prazole", "olol", "dipine",
              "ate", "ide", "gen", "toxin", "feron", "kin", "leukin")


def worth_single_word(word: str) -> bool:
    """单个词是否值得进术语表：缩写、或带医学术语/药类词尾。

    通用名词（weight / powder / ointment / response）不构成术语，会污染命中率。
    """
    w = word.lower()
    if word.isupper() and 2 <= len(word) <= 6:
        return True
    return any(w.endswith(s) and len(w) > len(s) + 2 for s in MED_SUFFIX)


def best_surface(counter: Counter) -> str:
    """同一 signature 的多个表面串里挑一个展示：优先非全大写、首字母大写的写法。"""
    if not counter:
        return ""
    ranked = sorted(counter.items(), key=lambda kv: (
        kv[1],                                   # 频次优先
        not kv[0][:1].isupper(),                 # 首字母大写更好
        kv[0].isupper(),                         # 全大写最差
    ), reverse=True)
    return ranked[0][0]



def content_tokens(span: str, side: str) -> list[str]:
    toks = (EN_TOK if side == "en" else FR_TOK).findall(span)
    out = []
    for t in toks:
        tl = t.lower().strip("'-")
        if not tl or NUM_RE.match(tl) or UNIT_RE.match(tl):
            continue
        if tl in STOP or tl in BOILER or (side == "fr" and tl in FR_STOP):
            continue
        out.append(tl)
    return out


def signature(span: str, side: str) -> tuple[str, ...]:
    ct = content_tokens(span, side)
    return tuple(sorted(set(ct))) if ct else ()


def good_span(span: str, side: str = "en") -> bool:
    """掐掉被切碎的短语（Nature and / de benzalkonium）与纯单位串。"""
    toks = span.lower().split()
    if not toks:
        return False
    if UNIT_RE.match(toks[0]) or UNIT_RE.match(toks[-1]):
        return False
    edge = EDGE | (FR_STOP if side == "fr" else set())
    return toks[0] not in edge and toks[-1] not in edge


def ngrams(tokens: list[str], n_max: int) -> list[str]:
    out = []
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            g = " ".join(tokens[i:i + n])
            if len(g) <= 60:
                out.append(g)
    return out


SEG_SPLIT = re.compile(r"[/(),;:\[\]{}|]|\s[-–—]\s")


def build_index(rows: list[dict], side: str, n_max: int = 3) -> dict:
    """按内容词 signature 聚合：{sig: {"pairs": set, "surfaces": Counter}}。

    n-gram 只在"分隔符切出的词段"内部生成。EMEA 里 `body weight/day`、`mg/ml`、
    `and/or` 这类用斜杠/括号连写的结构，一旦按整句分词就会拼出 `body weight day`
    这种语料里不存在的假短语（2,029 对含斜杠）。
    """
    tok_re = EN_TOK if side == "en" else FR_TOK
    idx: dict[tuple, dict] = defaultdict(lambda: {"pairs": set(), "surfaces": Counter()})
    for i, r in enumerate(rows):
        text = normalize_text(r["src"] if side == "en" else r["tgt"], side)
        for seg in SEG_SPLIT.split(text):
            toks = tok_re.findall(seg)
            for g in ngrams(toks, n_max):
                if not good_span(g, side):
                    continue
                sig = signature(g, side)
                if not sig:
                    continue
                if not any(len(t) >= 4 for t in sig):
                    continue
                idx[sig]["pairs"].add(i)
                idx[sig]["surfaces"][repair(g)] += 1    # 保留原始大小写，展示时再挑
    return idx


def mode_align(args) -> None:
    rows = load("emea.train.jsonl")
    print(f"[align] EMEA train {len(rows)} 对", flush=True)

    en_idx = build_index(rows, "en")
    fr_idx = build_index(rows, "fr")
    print(f"  en signature {len(en_idx)}，fr signature {len(fr_idx)}（已按内容词合并词序）",
          flush=True)

    en_sel = {s: d for s, d in en_idx.items() if args.min_cooc <= len(d["pairs"]) <= args.max_freq}
    fr_sel = {s: d for s, d in fr_idx.items() if args.min_cooc <= len(d["pairs"]) <= args.max_freq}
    print(f"  进入对齐: en {len(en_sel)}，fr {len(fr_sel)}", flush=True)

    pair_fr: dict[int, list[tuple]] = defaultdict(list)
    for s, d in fr_sel.items():
        for i in d["pairs"]:
            pair_fr[i].append(s)

    raw: list[dict] = []
    for en_sig, d in en_sel.items():
        en_pairs = d["pairs"]
        c = Counter()
        for i in en_pairs:
            c.update(pair_fr.get(i, ()))
        for fr_sig, cooc in c.items():
            # 长度不一致的对齐基本是"截断"（carbonic anhydrase || anhydrase），
            # 允许差 2 个内容词以防漏掉 "international non-proprietary name" 这类，
            # 但排序时按长度一致度打折，同一英文只保留一致度最高的法语。
            diff = abs(len(en_sig) - len(fr_sig))
            if diff > 2:
                continue
            cont_en = cooc / len(en_pairs)
            if cont_en < args.min_containment:
                continue
            cont_fr = cooc / len(fr_sel[fr_sig]["pairs"])
            score = min(cont_en, cont_fr)
            if score < args.min_containment:
                continue
            rank = score * (1.0 - 0.20 * diff) * (1.0 + 0.05 * (len(en_sig) + len(fr_sig)))
            raw.append({
                "en": clean_surface(best_surface(d["surfaces"])),
                "fr": clean_surface(best_surface(fr_sel[fr_sig]["surfaces"])),
                "en_sig": list(en_sig), "fr_sig": list(fr_sig),
                "en_variants": [s for s, _ in d["surfaces"].most_common(4)],
                "fr_variants": [s for s, _ in fr_sel[fr_sig]["surfaces"].most_common(4)],
                "cooc": cooc, "en_freq": len(en_pairs), "fr_freq": len(fr_sel[fr_sig]["pairs"]),
                "cont_en": round(cont_en, 3), "cont_fr": round(cont_fr, 3),
                "score": round(score, 3), "rank": round(rank, 3), "len_diff": diff,
                "pmi": round(math.log(cooc * len(rows) / (len(en_pairs) * len(fr_sel[fr_sig]["pairs"]))), 2),
            })
    print(f"  原始候选 {len(raw)}", flush=True)

    audit = {}

    # 四个谓词都是保留语义：返回 True 才留下
    def distinct_content(d):
        """内容词集合完全相同（词序/大小写不同）不是翻译，丢掉。"""
        return tuple(d["en_sig"]) != tuple(d["fr_sig"])

    def has_sig(d):
        return bool(d["en_sig"]) and bool(d["fr_sig"])

    def translated(d):
        if abs(len(d["en_sig"]) - len(d["fr_sig"])) > 2:
            return False
        from difflib import SequenceMatcher
        return SequenceMatcher(None, d["en"].lower(), d["fr"].lower()).ratio() < 0.82

    def not_boiler(d):
        en = d["en"]
        if en in BOILER_VERB:
            return False
        if any(t in BOILER_VERB for t in en.lower().split()):  # "solution Select" 这类
            return False
        if any(t in PDF_NOISE for t in en.split()):
            return False
        return not any(en.startswith(p) for p in BOILER_PREFIX)

    def clean_entry(d):
        if any(len(t) <= 1 for t in d["en_sig"]):             # 掩码占位 "x film-coated tablets"
            return False
        if len(d["en_sig"]) == 1 and not worth_single_word(d["en"]):   # 通用单词不是术语
            return False
        if looks_brand(d["en"]):                              # 商标/公司名
            return False
        fr_toks = content_tokens(d["fr"], "fr")
        if any(t in BOILER_VERB or t in FR_VERB for t in fr_toks):   # 混排/指令残片
            return False
        if any(BRAND_RE.match(t) for t in d["fr"].split()):          # 右侧带商标（ABILIFY 这类）
            return False
        return True

    for name, fn in (("词序/大小写相同（非翻译）", distinct_content),
                     ("纯单位或数字", has_sig),
                     ("直译雷同", translated),
                     ("说明书套话", not_boiler),
                     ("专名/掩码/混排残片", clean_entry)):
        before = len(raw)
        raw = [d for d in raw if fn(d)]
        audit[name] = before - len(raw)
        print(f"  过滤 {name}: -{audit[name]}", flush=True)

    # 排序：长度一致度打折后的 rank 优先，其次共现，其次总内容词数（信息量更大）
    raw.sort(key=lambda d: (-d["rank"], -d["cooc"],
                            -(len(d["en_sig"]) + len(d["fr_sig"]))))

    # 一侧多译只留最优：en_sig -> 最佳 fr_sig；fr_sig -> 最佳 en_sig
    best_by_en: dict[tuple, dict] = {}
    for d in raw:
        best_by_en.setdefault(tuple(d["en_sig"]), d)
    best_by_fr: dict[tuple, dict] = {}
    for d in sorted(best_by_en.values(),
                    key=lambda x: (-x["rank"], -x["cooc"], -len(x["fr_sig"]))):
        best_by_fr.setdefault(tuple(d["fr_sig"]), d)
    one2one = sorted(best_by_fr.values(),
                     key=lambda d: (-d["rank"], -d["cooc"],
                                    -(len(d["en_sig"]) + len(d["fr_sig"]))))
    audit["同英多译/同法多源去重"] = len(best_by_en) - len(one2one)
    print(f"  一侧多译去重: {len(best_by_en)} -> {len(one2one)}"
          f"（-{audit['同英多译/同法多源去重']}）", flush=True)

    # 5) 子集去重：X 被 X Y 覆盖时丢 X
    kept: list[dict] = []
    for d in one2one:
        covered = any(
            k is not d and
            (set(d["en_sig"]) < set(k["en_sig"]) or set(d["fr_sig"]) < set(k["fr_sig"])) and
            k["cooc"] >= d["cooc"] * 0.5
            for k in kept)
        if not covered:
            kept.append(d)
    audit["被更长词条覆盖的子集"] = len(one2one) - len(kept)
    print(f"  子集去重: {len(one2one)} -> {len(kept)}（-{audit['被更长词条覆盖的子集']}）", flush=True)

    kept = kept[:args.top]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    for d in kept:
        d["variants"] = []
        for variant in (d["fr"], d["fr"][:1].upper() + d["fr"][1:]):
            ids = tok(" " + variant, add_special_tokens=False)["input_ids"]
            d["variants"].append({"text": variant, "ids": ids, "n_tokens": len(ids),
                                  "single_token": len(ids) == 1})

    path = OUT / "glossary.emea.candidates.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for d in kept:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    single = sum(1 for d in kept if d["variants"][0]["single_token"])
    audit["输出条数"] = len(kept)
    audit["其中法语单 token"] = single
    (OUT / "glossary.emea.audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  输出 {len(kept)} 条（法语单 token {single} 条） -> {path}", flush=True)
    print("  过滤明细: " + json.dumps(audit, ensure_ascii=False), flush=True)
    for d in kept[:args.show]:
        print(f"  {d['score']:.2f} cooc={d['cooc']:>4} {d['en'][:40]:40s} || {d['fr'][:36]:36s}"
              f" tok={d['variants'][0]['n_tokens']}", flush=True)


def mode_audit(args) -> None:
    """体检现有候选表：单位、重复、词序、长度。"""
    path = OUT / "glossary.emea.candidates.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    print(f"[audit] {path.name}：{len(rows)} 条", flush=True)

    def has_unit(d, side):
        toks = (d["en"] if side == "en" else d["fr"]).split()
        return any(UNIT_RE.match(t.strip(".,")) for t in toks)

    print(f"  英文侧含单位/数字: {sum(1 for d in rows if has_unit(d, 'en'))}")
    print(f"  法语侧含单位/数字: {sum(1 for d in rows if has_unit(d, 'fr'))}")
    esig = Counter(tuple(d.get("en_sig") or signature(d["en"], "en")) for d in rows)
    fsig = Counter(tuple(d.get("fr_sig") or signature(d["fr"], "fr")) for d in rows)
    print(f"  英文 signature 重复组: {sum(1 for v in esig.values() if v > 1)}"
          f"（涉及 {sum(v for v in esig.values() if v > 1)} 条）")
    print(f"  法语 signature 重复组: {sum(1 for v in fsig.values() if v > 1)}"
          f"（涉及 {sum(v for v in fsig.values() if v > 1)} 条）")
    multi = [d for d in rows if len(d["fr"].split()) > 4]
    print(f"  法语侧 >4 词的超长条目: {len(multi)}")
    for d in multi[:5]:
        print(f"    {d['en'][:34]:34s} || {d['fr'][:50]}")
    if args.show:
        print("\n  前 20 条：")
        for d in rows[:args.show]:
            print(f"    {d['score']:.2f} {d['en'][:38]:38s} || {d['fr'][:36]}")


def mode_slang(args) -> None:
    """跨域对比：在 MTNT 里高频、在 EMEA/clean 里几乎不出现的短 token。"""
    mtnt = load("mtnt.train.jsonl")
    emea = load("emea.train.jsonl")
    clean = load("clean.train.jsonl")

    def counts(rows):
        c = Counter()
        for r in rows:
            c.update(t.lower() for t in EN_TOK.findall(r["src"]))
        return c

    cm, ce, cc = counts(mtnt), counts(emea), counts(clean)
    n_m = len(mtnt)

    FRAG = {"ll", "re", "ve", "s", "t", "d", "m", "don", "doesn", "isn", "wasn", "aren",
            "didn", "won", "can", "couldn", "wouldn", "shouldn", "haven", "hasn", "hadn"}
    SEED = {"tbh", "imo", "imho", "idk", "irl", "btw", "afaik", "smh", "ngl", "rn", "af",
            "lol", "lmao", "rofl", "wtf", "omg", "tmi", "fyi", "iirc", "op", "eli5",
            "tl", "dr", "til", "mfw", "tfw", "jk", "np", "ty", "gg", "ez", "asap",
            "dm", "pm", "ama", "nsfw", "ffs", "stfu", "gtfo", "ikr", "tbf", "ngl"}

    def is_slangish(tok_: str) -> bool:
        if tok_ in SEED:
            return True
        if tok_.upper() == tok_ and 2 <= len(tok_) <= 5:
            return True
        return not re.search(r"[aeiou]", tok_)

    cand = []
    for tok_, n in cm.items():
        if n < args.min_cooc or len(tok_) < 2 or tok_ in STOP or tok_ in FRAG:
            continue
        if not re.fullmatch(r"[a-z][a-z'\-]*", tok_):
            continue
        if not is_slangish(tok_):
            continue
        if (ce.get(tok_, 0) + cc.get(tok_, 0)) / n_m > 0.02:
            continue
        cand.append({"token": tok_, "mtnt_freq": n, "emea_freq": ce.get(tok_, 0),
                     "clean_freq": cc.get(tok_, 0),
                     "in_seed": tok_ in SEED})

    cand.sort(key=lambda d: (not d["in_seed"], -d["mtnt_freq"]))
    cand = cand[:args.top]

    path = OUT / "glossary.mtnt.candidates.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for d in cand:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[slang] MTNT train {n_m} 句 -> 候选 {len(cand)}（种子表内 "
          f"{sum(1 for d in cand if d['in_seed'])} 条）-> {path}", flush=True)
    for d in cand[:args.show]:
        print(f"  {d['token']:14s} mtnt={d['mtnt_freq']:>4} emea={d['emea_freq']:>3} "
              f"clean={d['clean_freq']:>3} {'[seed]' if d['in_seed'] else ''}", flush=True)


def mode_llm(args) -> None:
    """LLM 抽取（召回补充）：本地 Qwen 逐对抽术语，与统计结果交叉验证。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load("emea.train.jsonl")
    rng = __import__("random").Random(42)
    sample = rng.sample(rows, min(args.samples, len(rows)))
    print(f"[llm] 抽 {len(sample)} 对做术语抽取", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok.padding_side = "left"
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()

    SYS = ("You extract bilingual terminology from a pharmaceutical document. "
           "Return ONLY a JSON array, no prose. Each item: "
           '{"en": "<English term>", "fr": "<French term>"}. '
           "Keep only domain-specific terms: diseases, drug classes, anatomy, "
           "regulatory or pharmaceutical procedures. Exclude brand names, company "
           "names, units and doses (mg, ml, IU), page numbers, and ordinary vocabulary. "
           "At most 4 items.")

    def prompt(r):
        return f"{SYS}\n\nEN: {r['src'][:400]}\nFR: {r['tgt'][:400]}\n\nJSON:"

    counts: Counter = Counter()
    fails = 0

    def parse(text: str) -> list[dict]:
        i, j = text.find("["), text.rfind("]")
        if i >= 0 and j > i:
            try:
                obj = json.loads(text[i:j + 1])
                return [d for d in obj if isinstance(d, dict) and d.get("en") and d.get("fr")]
            except Exception:  # noqa: BLE001
                pass
        return [{"en": m.group(1), "fr": m.group(2)} for m in
                re.finditer(r'"en"\s*:\s*"([^"]{2,50})"\s*,\s*"fr"\s*:\s*"([^"]{2,60})"', text)]

    for start in range(0, len(sample), 16):
        chunk = sample[start:start + 16]
        enc = tok([prompt(r) for r in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            pairs = parse(t)
            if not pairs:
                fails += 1
            for d in pairs:
                counts[(d["en"].strip().lower(), d["fr"].strip().lower())] += 1
        print(f"  {start + len(chunk)}/{len(sample)}", flush=True)

    ranked = [{"en": en, "fr": fr, "votes": v} for (en, fr), v in counts.most_common()
              if not any(UNIT_RE.match(t) for t in en.split())]
    path = OUT / "glossary.emea.llm.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for d in ranked:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[llm] 抽到 {len(ranked)} 条去重术语，解析失败 {fails}/{len(sample)} -> {path}",
          flush=True)

    stat_path = OUT / "glossary.emea.candidates.jsonl"
    if stat_path.exists():
        stat = [json.loads(l) for l in stat_path.open(encoding="utf-8")]
        stat_en = {tuple(d.get("en_sig") or signature(d["en"], "en")) for d in stat}
        hit = sum(1 for d in ranked if signature(d["en"], "en") in stat_en)
        print(f"[cross] LLM 术语的英文 signature 与统计表重合 {hit}/{len(ranked)}", flush=True)


def mode_coverage(args) -> None:
    emea_terms = [json.loads(l) for l in
                  (OUT / "glossary.emea.candidates.jsonl").open(encoding="utf-8")][:args.top]
    mtnt_terms = [json.loads(l) for l in
                  (OUT / "glossary.mtnt.candidates.jsonl").open(encoding="utf-8")][:args.top]

    print(f"[coverage] EMEA {len(emea_terms)} 条 / MTNT {len(mtnt_terms)} 条候选", flush=True)
    for split in ("dev", "test"):
        rows = load(f"emea.{split}.jsonl")
        pats = [re.compile(re.escape(t["en"].lower())) for t in emea_terms]
        covered = sum(1 for r in rows if any(p.search(r["src"].lower()) for p in pats))
        print(f"  emea.{split}: {covered}/{len(rows)} 句"
              f"（{100 * covered / len(rows):.1f}%）", flush=True)
    for split in ("dev", "test"):
        rows = load(f"mtnt.{split}.jsonl")
        pats = [re.compile(r"(^|\W)" + re.escape(t["token"].lower()) + r"(\W|$)")
                for t in mtnt_terms]
        covered = sum(1 for r in rows if any(p.search(r["src"].lower()) for p in pats))
        print(f"  mtnt.{split}: {covered}/{len(rows)} 句"
              f"（{100 * covered / len(rows):.1f}%）", flush=True)


def mode_prune(args) -> None:
    """按"评测可达性"剪枝：dev 与 test 都不出现的条目对约束解码和命中率都无贡献。

    保留全量到 *.full.jsonl 以便需要时回滚。
    """
    path = OUT / "glossary.emea.candidates.jsonl"
    full = OUT / "glossary.emea.candidates.full.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not full.exists():
        full.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")

    dev = [r["src"].lower() for r in load("emea.dev.jsonl")]
    test = [r["src"].lower() for r in load("emea.test.jsonl")]

    kept, dropped = [], 0
    for r in rows:
        en = r["en"].lower()
        r["dev_hits"] = sum(1 for s in dev if en in s)
        r["test_hits"] = sum(1 for s in test if en in s)
        if r["dev_hits"] + r["test_hits"] == 0:
            dropped += 1
            continue
        kept.append(r)

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prune] {len(rows)} -> {len(kept)}（剔除 dev/test 都不出现的 {dropped} 条）",
          flush=True)
    print(f"  全量备份: {full.name}", flush=True)
    print(f"  其中 test 有命中 {sum(1 for r in kept if r['test_hits'] > 0)} 条，"
          f"仅 dev 命中 {sum(1 for r in kept if r['test_hits'] == 0)} 条", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["align", "slang", "llm", "coverage", "audit", "prune"],
                    required=True)
    ap.add_argument("--min-cooc", type=int, default=6, dest="min_cooc")
    ap.add_argument("--max-freq", type=int, default=300, dest="max_freq")
    ap.add_argument("--min-containment", type=float, default=0.5, dest="min_containment")
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()
    {"align": mode_align, "slang": mode_slang, "llm": mode_llm,
     "coverage": mode_coverage, "audit": mode_audit, "prune": mode_prune}[args.mode](args)


if __name__ == "__main__":
    main()
