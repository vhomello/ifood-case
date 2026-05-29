# Ifood Case

## Project structure:

```text
ifood-case/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 1_data_processing.ipynb
│   └── 2_modeling.ipynb
├── presentation/
├── src/
├── README.md
└── requirements.txt
```

## Setup
This project uses `uv` for package and environment management
The file `pyproject.toml` and `uv.lock` are used to manage dependencies and `requirements.txt` was generated using the following command
```
uv export --format requirements-txt -o requirements.txt
```

### Package Installation
You can install the codebase as an editable package inside your environment by running:
```bash
pip install -e .
```

This registers the package, allowing you to import modules of `ifood` in your notebooks and Python scripts:
```python
from ifood.data_processing.profile import read_profile_json
```