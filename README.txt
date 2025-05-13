# CTMS Plan Symbol-to-KMZ Toolkit

This package includes the starting point for your ITS symbol detection workflow.

Included:
- `extract_symbols_from_legend.py`: Run this to select and save legend symbols as templates.

Folders created:
- `templates/existing/` — for existing ITS feature templates
- `templates/proposed/` — for proposed ITS feature templates

Next Step:
Use `match_and_generate_kmz.py` (coming next) to find matches and build your KMZ.

You'll need:
- Python 3.x
- pip install pymupdf opencv-python numpy

