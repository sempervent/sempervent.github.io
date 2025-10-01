# sempervent.github.io
Sempervent Landing Page

This repository uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) to generate documentation.

## Quick Start

### Local Development

1. Install MkDocs and dependencies:
   ```bash
   pip install mkdocs mkdocs-material
   ```

2. Run the development server:
   ```bash
   mkdocs serve
   ```

3. View the site at `http://127.0.0.1:8000`

### Building the Site

To build the static site:
```bash
mkdocs build
```

The generated site will be in the `site/` directory.

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch using GitHub Actions.

The workflow is defined in `.github/workflows/deploy-mkdocs.yml`.

## Project Structure

```
.
├── docs/                    # Documentation source files
│   ├── index.md            # Home page
│   ├── about.md            # About page
│   ├── getting-started.md  # Getting started guide
│   └── documentation.md    # Documentation page
├── mkdocs.yml              # MkDocs configuration
└── .github/
    └── workflows/
        └── deploy-mkdocs.yml  # GitHub Actions workflow
```

## Customization

Edit the following files to customize your site:

- `mkdocs.yml` - Site configuration, theme, navigation
- `docs/*.md` - Page content (Markdown format)

## License

See [LICENSE](LICENSE) file for details.
