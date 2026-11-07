# Devito and PyTorch articles

- [Quick read](devito-pytorch-short.md): the network layer and the core API.
- [Full walkthrough](devito-pytorch.md): discretization, discrete adjoint,
  training, tensor ownership, generated code, OpenACC and MPI.

Both track the backend in `feature/torch-backend`. The article sources live on
`docs/torch-blog`; rendered site files live on the personal fork's `gh-pages`.

Render both versions with Python, Markdown and Pygments installed:

```sh
python docs/blog/render.py
```

Each HTML file embeds its styles and SVG figures and works without JavaScript,
a CDN or a build server. Keep both files together for the edition links.
`index.html` is the quick-read landing page.

The training loss CSV and figures accompany the reproducible CPU example in
`examples/pytorch/devito_layer.py`. The CPU code snippets were executed and both
editions were checked in Chromium at desktop and mobile widths. NVIDIA execution
is covered separately by the backend's existing GPU workflow.
