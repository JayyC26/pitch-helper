"""Rebuild index.html's inline script from src/analysis.js + src/app.js (single-file app)."""
import re, pathlib
here = pathlib.Path(__file__).parent
ana = (here / "src/analysis.js").read_text(); app = (here / "src/app.js").read_text()
names = re.findall(r'^export (?:const|function|class) (\w+)', ana, flags=re.M)
for line in re.findall(r'^export const (.+);$', ana, flags=re.M):
    names += re.findall(r'(\b[A-Z_]{2,}\w*)\s*=', line)
names = sorted(set(names))
js = ("const A = (() => {\n" + re.sub(r'^export ', '', ana, flags=re.M) + "\nreturn { " + ", ".join(names) + " };\n})();\n\n"
      + re.sub(r'^import \* as A from "\./analysis\.js";\n', '', app, flags=re.M))
assert "</script" not in js.lower()
html = (here / "index.html").read_text()
start = html.index('<script type="module">'); end = html.index('</script>', start) + len('</script>')
html = html[:start] + '<script type="module">\n// Everything is inlined in this one file so it also works in a private Hugging Face Space.\n' + js + '\n</script>' + html[end:]
(here / "index.html").write_text(html)
print("built index.html", len(html) // 1024, "KB")
