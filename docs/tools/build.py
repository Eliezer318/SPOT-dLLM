#!/usr/bin/env python
"""Build the project page: inline assets/traces.json into src/page.html.

    python docs/tools/build.py                       # writes docs/index.html (full HTML, for GitHub Pages)
    python docs/tools/build.py --fragment OUT.html   # the page without <html>/<head>/<body>, for previews

src/page.html holds the <head> content, a `<!--BODY-->` marker, then the body.
"""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
page = open(os.path.join(D, "src", "page.html"), encoding="utf-8").read()
data = json.dumps(json.load(open(os.path.join(D, "assets", "traces.json"))), separators=(",", ":"), ensure_ascii=False)
page = page.replace("/*__TRACES__*/", data.replace("</", "<\\/"))
head, body = page.split("<!--BODY-->", 1)
if "--fragment" in sys.argv:
    out = sys.argv[sys.argv.index("--fragment") + 1]
    # previews keep a short tab title; the published page carries the full paper title
    import re
    head = re.sub(r"<title>[^<]*</title>", "<title>On the SPOT</title>", head, count=1)
    open(out, "w", encoding="utf-8").write(head + body)
else:
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
    out = os.path.join(D, "index.html")
    open(out, "w", encoding="utf-8").write(full)
print("wrote", out, os.path.getsize(out), "bytes")
