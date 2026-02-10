# CLAUDE.md

Guide for AI assistants working on this repository.

## Project Overview

Personal knowledge-base and documentation site by Murtaza Nazir, covering AI/ML topics (linear algebra, neural networks). Built with **MkDocs** and the **Material for MkDocs** theme, deployed to GitHub Pages at https://themurtazanazir.github.io/.

## Repository Structure

```
.
├── mkdocs.yml                  # Main site configuration (theme, nav, plugins, extensions)
├── docs/                       # All site content lives here
│   ├── index.md                # Home / landing page
│   ├── img/                    # Site-wide images (logo.svg, photo.jpg)
│   ├── stylesheets/
│   │   └── extra.css           # Custom CSS overrides (~270 lines)
│   ├── linear_algebra/         # Linear algebra notes (3 pages)
│   ├── neural_networks/        # Neural network notes with subsections
│   │   ├── multiayer-perceptron/   # Note: typo "multiayer" is intentional — do NOT rename
│   │   ├── convolutional_neural_networks/
│   │   └── transformer/
│   └── blog/                   # Blog (not active yet, listed in .gitignore)
├── .github/workflows/ci.yml   # CI/CD pipeline
├── .gitignore
└── README.md
```

## Build & Development

### Prerequisites

Python 3.x with the following packages:

```bash
pip install mkdocs-material mkdocs-roamlinks-plugin mkdocs-rss-plugin
```

### Local Development

```bash
mkdocs serve
```

This starts a local dev server with live reload (default: http://127.0.0.1:8000).

### Deployment

Deployment is **automatic** — every push to any branch triggers the GitHub Actions workflow at `.github/workflows/ci.yml`, which runs `mkdocs gh-deploy --force`. No manual deploy steps are needed.

## Content Conventions

### Adding New Pages

1. Create a `.md` file under the appropriate `docs/` subdirectory.
2. Add the page to the `nav:` section in `mkdocs.yml` to include it in navigation.
3. Use YAML front matter for `title` and `description` when appropriate.

### Markdown Features Available

- **LaTeX math**: Use `$...$` for inline and `$$...$$` for display math (rendered via MathJax 3).
- **Mermaid diagrams**: Use fenced code blocks with language `mermaid`.
- **Wiki-style links**: Use `[[filename|display text]]` syntax (via `roamlinks` plugin).
- **Task lists**: `- [ ]` / `- [x]` syntax.
- **Footnotes**: `[^1]` syntax.
- **Critic markup**, **caret**, **mark**, **tilde**, **keys** extensions from PyMdown.

### Images and Media

- Site-wide assets go in `docs/img/`.
- Section-specific figures go in subdirectories near their content (e.g., `docs/neural_networks/convolutional_neural_networks/Figures/`).
- MP4 video files are used for animations in some sections.

### Styling

The site uses a heavily customized minimal design defined in `docs/stylesheets/extra.css`:
- Navigation UI (header, sidebars, footer) is **hidden via CSS** for a clean reading experience.
- Warm color palette: beige background (`#fef6ec`), brown text (`#4a3f35`), blue links (`#0066cc`).
- Serif font stack (`ui-serif, Georgia, Cambria, "Times New Roman"`).
- Content is centered with `max-width: 48rem`.
- No animations or transitions.
- `.drop-cap` class available for decorative first letters.

When modifying styles, maintain this minimal, academic aesthetic. Do not reintroduce Material theme chrome (shadows, sidebars, animations).

## Configuration Reference

Key settings in `mkdocs.yml`:

| Setting | Value |
|---------|-------|
| Theme | `material` (brown primary, deep orange accent) |
| TOC | Disabled (`toc_depth: 0`, `permalink: false`) |
| Search | Enabled |
| Blog plugin | Configured but content gitignored |
| Code copy button | Enabled (`content.code.copy`) |

## CI/CD

The pipeline (`.github/workflows/ci.yml`) runs on every push:
1. Checks out code
2. Sets up Python 3.x
3. Caches `.cache` directory for faster builds
4. Installs dependencies via pip
5. Deploys with `mkdocs gh-deploy --force`

There are no tests, linters, or pre-commit hooks configured.

## Important Notes

- The `docs/blog/` directory is gitignored and not in use yet — do not add blog content without explicit instruction.
- The directory `multiayer-perceptron` contains a typo (missing "l" in "multilayer"). This is established in navigation URLs and links — do **not** rename it.
- There is no `requirements.txt` — dependencies are installed directly in CI. If adding a new MkDocs plugin, update both `mkdocs.yml` and `.github/workflows/ci.yml`.
- The `.obsidian` directory is gitignored — content may be authored in Obsidian, so wiki-style `[[links]]` are used.
