## Pre-requisites

You need Python >=3.11 on your machine to install `polars-io-tools`.

## Install with `pip`

```bash
pip install polars-io-tools
```

The base install is intentionally lightweight. Feature-specific integrations
(SQL/ODBC readers, Delta, ClickHouse, Datadog, S3 caching, Ray execution, the
synthetic data generators, and the narwhals reader) rely on optional
third-party packages that are bundled in the `full` extra:

```bash
pip install "polars-io-tools[full]"
```

Calling a feature whose optional dependency is missing raises a clear error
telling you to install `polars-io-tools[full]`.

## Install with `conda`

```bash
conda install polars-io-tools --channel conda-forge
```

## Source installation

For other platforms and for development installations, [build `polars-io-tools` from source](Build-from-Source).
