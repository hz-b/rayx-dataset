# RAYX-Dataset

A dataset generation tool for RAYX.

## Quick Start
### Podman
`podman build -t rayx-dataset:latest . `
`podman run --rm -it --name test --replace rayx-dataset:latest python3 generate.py seed=42`
