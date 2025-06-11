<<<<<<< HEAD
# Dependency Management

## Canonical Source: pyproject.toml

Use `pyproject.toml` to define:

- Project name, version, authors
- Dependencies (and dev-dependencies)
- Tool configs (`black`, `mypy`, `isort`...)

## Docker/CI Compatibility

Generate requirements:
```bash
poetry export -f requirements.txt --without-hashes > requirements.txt
```

## Syncing

To avoid drift:
- Lock dependencies locally (`poetry.lock`)
- Export for CI (`requirements.txt`)
- Use `Makefile` targets:
```makefile
requirements.txt: pyproject.toml
	poetry export -f requirements.txt --without-hashes > requirements.txt
```
=======
# Dependency Management

## Canonical Source: pyproject.toml

Use `pyproject.toml` to define:

- Project name, version, authors
- Dependencies (and dev-dependencies)
- Tool configs (`black`, `mypy`, `isort`...)

## Docker/CI Compatibility

Generate requirements:
```bash
poetry export -f requirements.txt --without-hashes > requirements.txt
```

## Syncing

To avoid drift:
- Lock dependencies locally (`poetry.lock`)
- Export for CI (`requirements.txt`)
- Use `Makefile` targets:
```makefile
requirements.txt: pyproject.toml
	poetry export -f requirements.txt --without-hashes > requirements.txt
```
>>>>>>> 0d87d2ef0905c76b1b409a597d20537b5f24502e
