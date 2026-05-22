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
The file `pyproject.toml` and `uv.lock` is used to manage dependencies and `requirements.txt` was using the following command
```
uv export --format requirements-txt -o requirements.txt
```