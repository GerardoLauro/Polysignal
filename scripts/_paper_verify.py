import re, os
from collections import Counter
os.chdir(os.path.join(os.path.dirname(__file__), "..", "pdf_latex"))
texs = ["main.tex", "part1_foundations.tex", "part2_pipeline.tex",
        "part3_analysis.tex", "part4_prediction.tex"]
ok = True

# 1) non-ASCII
for t in texs:
    for i, line in enumerate(open(t, encoding="utf-8").read().splitlines(), 1):
        bad = [(c, hex(ord(c))) for c in line if ord(c) > 127]
        if bad:
            print(f"[NONASCII] {t}:{i} {bad} :: {line.strip()[:60]}"); ok = False

# 2) begin/end balance
for t in texs:
    s = open(t, encoding="utf-8").read()
    cb = Counter(re.findall(r"\\begin\{([^}]+)\}", s))
    ce = Counter(re.findall(r"\\end\{([^}]+)\}", s))
    for env in set(cb) | set(ce):
        if cb[env] != ce[env]:
            print(f"[ENVDIFF] {t}: {env} begin={cb[env]} end={ce[env]}"); ok = False

# 3) brace balance (ignore escaped braces)
for t in texs:
    s = open(t, encoding="utf-8").read().replace("\\{", "").replace("\\}", "")
    o, c = s.count("{"), s.count("}")
    if o != c:
        print(f"[BRACE] {t}: open={o} close={c} diff={o - c}"); ok = False

# 4) includegraphics files exist
figs = set()
for t in texs:
    figs |= set(re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}",
                           open(t, encoding="utf-8").read()))
for f in sorted(figs):
    if not os.path.exists(os.path.join("figures", f)):
        print(f"[MISSINGFIG] {f}"); ok = False
print(f"figures referenced={len(figs)} present="
      f"{sum(os.path.exists(os.path.join('figures', f)) for f in figs)}")

# 5) cite keys vs bib
keys = set(re.findall(r"@\w+\{([^,]+),", open("references.bib", encoding="utf-8").read()))
cited = set()
for t in texs:
    for grp in re.findall(r"\\cite\{([^}]+)\}", open(t, encoding="utf-8").read()):
        cited |= {k.strip() for k in grp.split(",")}
missing = cited - keys
if missing:
    print("[BADCITE]", missing); ok = False
print("cite used=%d defined=%d unused=%s" % (len(cited), len(keys), sorted(keys - cited)))

# 6) labels vs refs
labels = set()
refs = set()
for t in texs:
    s = open(t, encoding="utf-8").read()
    labels |= set(re.findall(r"\\label\{([^}]+)\}", s))
    refs |= set(re.findall(r"\\[Cc]ref\{([^}]+)\}", s))
refs = {r.strip() for grp in refs for r in grp.split(",")}
broken = refs - labels
if broken:
    print("[BROKENREF]", sorted(broken)); ok = False
print("labels=%d refs=%d" % (len(labels), len(refs)))

print("\nRESULT:", "ALL CHECKS PASS" if ok else "ISSUES FOUND")
