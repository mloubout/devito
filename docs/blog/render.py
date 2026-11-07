"""Render the blog as standalone HTML (requires Markdown and Pygments)."""

import re
from html import escape
from math import ceil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'devito-pytorch.md').read_text()
source = re.sub(
    r'\\\[.*?\\\]',
    '<div class="equation" aria-label="Diffusion equation">'
    '<span class="fraction"><span>∂u</span><span>∂t</span></span>'
    '<span>= α∇²u</span></div>', source, flags=re.S
)
renderer = markdown.Markdown(extensions=['fenced_code', 'codehilite', 'toc'],
                             extension_configs={'codehilite': {
                                 'css_class': 'highlight', 'guess_lang': False}})
body = renderer.convert(source)
title = re.search(r'^# (.*)$', source, re.M).group(1)
header = ('<header><div class="eyebrow">DEVITO × PYTORCH · DEVELOPMENT NOTES</div>'
          f'<h1>{escape(title)}</h1><div class="reading-time">'
          f'{ceil(len(source.split())/200)} minute read · '
          'CPU forward and backward validated</div></header>')
body = re.sub(r'<h1[^>]*>.*?</h1>', header, body, count=1, flags=re.S)


def embed_svg(match):
    attributes = match.group(0)
    filename = re.search(r'src="([^"]+)"', attributes).group(1)
    alternative = re.search(r'alt="([^"]*)"', attributes)
    svg = (ROOT / filename).read_text()
    svg = re.sub(r'<\?xml.*?\?>|<!DOCTYPE.*?>', '', svg, flags=re.S)
    return '<figure role="img" aria-label="{}">{}</figure>'.format(
        alternative.group(1) if alternative else '', svg)


body = re.sub(r'<img\b[^>]*src="[^"]+\.svg"[^>]*>', embed_svg, body)
body = re.sub(r'<p>(<figure\b.*?</figure>)</p>', r'\1', body, flags=re.S)
navigation = ''.join('<a href="#{}">{}</a>'.format(entry['id'], entry['name'])
                     for section in renderer.toc_tokens if section['level'] == 1
                     for entry in section['children'])
style = (ROOT / 'style.css').read_text()
document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><style>{style}</style></head>'
            '<body><div class="layout">'
            f'<aside><strong>In this article</strong>{navigation}</aside><article>{body}'
            '<footer>Development example · CPU input gradients validated against '
            'PyTorch and finite differences. GPU runtime validation remains '
            'outstanding.</footer></article></div></body></html>')
output = ROOT / 'devito-pytorch.html'
output.write_text('\n'.join(line.rstrip() for line in document.splitlines()) + '\n')
print(output)
