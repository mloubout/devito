"""Render both articles as standalone HTML (requires Markdown and Pygments)."""

import re
from html import escape
from math import ceil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
EDITIONS = ('devito-pytorch-short', 'devito-pytorch')


def embed_svg(match):
    attributes = match.group(0)
    filename = re.search(r'src="([^"]+)"', attributes).group(1)
    alternative = re.search(r'alt="([^"]*)"', attributes)
    svg = (ROOT / filename).read_text()
    svg = re.sub(r'<\?xml.*?\?>|<!DOCTYPE.*?>', '', svg, flags=re.S)
    return '<figure role="img" aria-label="{}">{}</figure>'.format(
        alternative.group(1) if alternative else '', svg)


def render(name):
    source = (ROOT / f'{name}.md').read_text()
    reading_time = ceil(len(source.split()) / 200)
    source = re.sub(
        r'\\\[.*?\\\]',
        '<div class="equation" aria-label="Diffusion equation">'
        '<span class="fraction"><span>∂u</span><span>∂t</span></span>'
        '<span>= α∇²u</span></div>', source, flags=re.S
    )
    renderer = markdown.Markdown(
        extensions=['fenced_code', 'codehilite', 'toc'],
        extension_configs={'codehilite': {
            'css_class': 'highlight', 'guess_lang': False}}
    )
    body = renderer.convert(source)
    title = re.search(r'^# (.*)$', source, re.M).group(1)
    edition = 'QUICK READ' if name.endswith('-short') else 'FULL WALKTHROUGH'
    links = ''.join(
        f'<a href="{item}.html"'
        f'{" aria-current=page" if item == name else ""}>{label}</a>'
        for item, label in zip(EDITIONS, ('Quick read', 'Full walkthrough'), strict=True)
    )
    header = (
        '<header><div class="eyebrow">DEVITO × PYTORCH · '
        f'{edition}</div><h1>{escape(title)}</h1>'
        f'<div class="reading-time">{reading_time} minute read · Development branch</div>'
        f'<nav class="editions" aria-label="Article versions">{links}</nav></header>'
    )
    body = re.sub(r'<h1[^>]*>.*?</h1>', header, body, count=1, flags=re.S)
    body = re.sub(r'<img\b[^>]*src="[^"]+\.svg"[^>]*>', embed_svg, body)
    body = re.sub(r'<p>(<figure\b.*?</figure>)</p>', r'\1', body, flags=re.S)
    for item in EDITIONS:
        body = body.replace(f'href="{item}.md"', f'href="{item}.html"')
    labels = iter(re.findall(r'^```(\w+)', source, re.M))

    def codeblock(match):
        language = next(labels)
        label = {'python': 'Python', 'sh': 'Shell', 'cpp': 'C++', 'text': 'Data layout'}
        return ('<div class="codeblock"><div class="code-label">'
                f'{label.get(language, escape(language))}</div>{match.group(0)}</div>')

    body = re.sub(r'<div class="highlight">.*?</div>', codeblock, body, flags=re.S)
    navigation = ''.join(
        f'<a href="#{entry["id"]}">{escape(entry["name"])}</a>'
        for section in renderer.toc_tokens if section['level'] == 1
        for entry in section['children']
    )
    style = (ROOT / 'style.css').read_text()
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)}</title><style>{style}</style></head>'
        '<body><div class="layout">'
        f'<aside><strong>In this article</strong>{navigation}</aside><article>{body}'
        '<footer>Development example · The layer supplies a CPU first-order input '
        'adjoint. Backend CUDA interoperability is tested separately. '
        '<a href="https://github.com/devitocodes/devito/tree/feature/torch-backend">'
        'Backend source</a></footer></article></div></body></html>'
    )
    output = ROOT / f'{name}.html'
    output.write_text('\n'.join(line.rstrip() for line in document.splitlines()) + '\n')
    if name == EDITIONS[0]:
        (ROOT / 'index.html').write_text(output.read_text())
    print(output)


if __name__ == '__main__':
    for name in EDITIONS:
        render(name)
