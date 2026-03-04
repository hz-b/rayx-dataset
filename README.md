# RAYX-Dataset

Containerized dataset generation tool for RAYX.

## Build

```bash
podman build -t rayx-dataset:latest .
```

## Create Output Directory

```bash
mkdir -p outputs
```

## Run Dataset Generation

```bash
podman run --rm -it \
  --name rayx-dataset \
  --replace \
  -v ./outputs:/App/outputs:Z \
  rayx-dataset:latest \
  python3 generate.py seed=42
```

Generated datasets and logs will be written to `./outputs`.

## Use a Custom RML File

```bash
podman run --rm -it \
  --name rayx-dataset \
  --replace \
  -v ./outputs:/App/outputs:Z \
  -v ./beamline.rml:/App/conf/beamline.rml:Z \
  rayx-dataset:latest \
  python3 generate.py seed=42
```

Make sure `beamline.rml` exists before running the command.
