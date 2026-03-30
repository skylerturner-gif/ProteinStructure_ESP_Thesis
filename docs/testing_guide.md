# Testing Guide

## Running Tests

```bash
conda activate protein_esp
pip install pytest pytest-cov   # one-time

pytest tests/ -v
pytest tests/ --cov=src --cov-report=html   # with coverage
open htmlcov/index.html
```

## What to Test (and What Not To)

This project wraps external compiled tools (APBS, PDB2PQR, MSMS) and operates on large protein data files. The testing strategy is:

**Do test:**
- Pure Python logic: path construction (`ProteinPaths`), metadata I/O, protein filtering
- Parsing functions: PQR/PDB atom parsers, xyzr extraction, `.dx` grid readers
- Math: interpolation, Laplacian reconstruction, Pearson r / RMSE computation
- Thread-safety of metadata updates

**Do not test in CI:**
- Subprocess calls to APBS, PDB2PQR, or MSMS — mock these instead
- Real protein files — use small synthetic fixtures instead
- The full pipeline end-to-end — verify this manually with a test protein

## Recommended File Structure

```
tests/
  conftest.py            # shared fixtures
  test_paths.py          # ProteinPaths construction and path logic
  test_io.py             # metadata create / update / load (thread-safety)
  test_filter.py         # protein ID filtering logic
  test_mesh_parsing.py   # xyzr_from_pqr, xyzr_from_pdb (no MSMS call)
  test_esp_mapping.py    # interpolation and Laplacian math
  test_graph_builder.py  # (future) graph construction logic
```

## Key Fixtures (`conftest.py`)

```python
import pytest
from pathlib import Path
from unittest.mock import patch

@pytest.fixture
def tmp_data_root(tmp_path):
    """Temporary data_root with a fake protein directory structure."""
    protein_id = "AF-TEST-F1"
    for subdir in ("structure", "electrostatics", "mesh", "esp", "logs"):
        (tmp_path / protein_id / subdir).mkdir(parents=True)
    return tmp_path

@pytest.fixture
def minimal_pqr_file(tmp_path):
    """Three-atom PQR file for testing parsers."""
    content = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000 -0.47  1.65\n"
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  0.07  1.87\n"
        "ATOM      3  C   ALA A   1       3.000   4.000   5.000  0.51  1.76\n"
    )
    pqr = tmp_path / "test.pqr"
    pqr.write_text(content)
    return pqr

@pytest.fixture
def mock_config(monkeypatch, tmp_data_root):
    """Patches get_config() so tests don't need a real config.yaml."""
    fake_cfg = {
        "paths": {"data_root": str(tmp_data_root), "log_file": str(tmp_data_root / "pipeline.log")},
        "executables": {"pdb2pqr": "pdb2pqr", "apbs": "apbs", "msms": "msms"},
        "electrostatics": {"forcefield": "PARSE", "ph_method": "propka", "ph": 7.0},
        "surface": {"msms_density": 3.0, "probe_radius": 1.4},
        "logging": {"level": "DEBUG"},
    }
    monkeypatch.setattr("src.utils.config.get_config", lambda: fake_cfg)
    return fake_cfg
```

## Mocking External Tools

Use `monkeypatch` to simulate APBS success or failure without invoking the binary:

```python
from types import SimpleNamespace

def test_apbs_failure_is_handled(monkeypatch, tmp_data_root, mock_config):
    monkeypatch.setattr(
        "src.electrostatics.run_apbs.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stderr="APBS error", stdout=""),
    )
    from src.electrostatics.run_apbs import process_apbs
    result = process_apbs("AF-TEST-F1", tmp_data_root)
    assert result is False
```

## Scientific Validation (Notebooks, Not CI)

These checks should be documented in `notebooks/` and run manually — not in CI:

- **ESP interpolation quality:** Pearson r between APBS grid values and interpolated surface ESP should be > 0.9 for well-folded proteins (mean pLDDT > 80)
- **Vertex count sanity:** vertex count should scale roughly with SES area at the configured `msms_density` (3.0 vertices/Å²)
- **Metadata completeness:** after running the full pipeline on a test protein, all expected metadata fields should be populated with no `None` values
