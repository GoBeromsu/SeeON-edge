# ruff: noqa: E501
"""Render a golden worksheet CSV as a single offline HTML labelling page.

The page plays each candidate clip from the local clip store, lets the owner
pick ``real`` / ``false`` / ``unsure`` per episode, and exports the filled
worksheet CSV from the browser (no server, no upload). The exported CSV is the
input to ``tests_support/golden_episodes.py``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Golden episode labelling</title>
<style>
body{{font-family:sans-serif;margin:16px}} .ep{{border:1px solid #ccc;padding:8px;margin:8px 0}}
video{{width:480px;height:270px;background:#000}} .row{{display:flex;gap:16px;align-items:flex-start}}
label{{margin-right:12px}} #bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #999}}
</style>
<div id="bar"><b id="count"></b> <button onclick="exportCsv()">CSV 내보내기</button>
<span>라벨: real = 실제 낙상/이탈, false = 오탐, unsure = 판단 불가</span></div>
<div id="list"></div>
<script>
const rows = {rows};
const fields = {fields};
const list = document.getElementById('list');
rows.forEach((r, i) => {{
  const d = document.createElement('div'); d.className = 'ep';
  d.innerHTML = `<div class="row"><video controls preload="none" src="file://${{r.clip_path}}"></video>
   <div><b>#${{i+1}}</b> ${{r.event_type}} · ${{r.camera_id.slice(0,8)}} · ${{r.detected_at}} · incidents ${{r.incident_count}}<br>
   ${{['real','false','unsure'].map(v => `<label><input type="radio" name="l${{i}}" value="${{v}}" ${{r.label===v?'checked':''}}> ${{v}}</label>`).join('')}}
   </div></div>`;
  d.querySelectorAll('input').forEach(el => el.onchange = () => {{ r.label = el.value; update(); }});
  list.appendChild(d);
}});
function update() {{ document.getElementById('count').textContent = rows.filter(r => r.label).length + ' / ' + rows.length + ' labelled'; }}
function exportCsv() {{
  const esc = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const text = [fields.join(',')].concat(rows.map(r => fields.map(f => esc(r[f])).join(','))).join('\\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {{type: 'text/csv'}}));
  a.download = 'golden-worksheet-labelled.csv'; a.click();
}}
update();
</script>
"""


def render(worksheet: Path, output: Path, labeller: str) -> int:
    if not labeller.strip():
        raise ValueError("labeller must not be empty")
    with worksheet.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "labeller" not in fields:
        fields.append("labeller")
    for row in rows:
        row["labeller"] = labeller
    page = _PAGE.format(
        rows=json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"),
        fields=json.dumps(fields),
    )
    output.write_text(page, encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=html.unescape(__doc__ or ""))
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labeller", required=True)
    args = parser.parse_args()
    print(f"episodes={render(args.worksheet, args.out, args.labeller)} html={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
