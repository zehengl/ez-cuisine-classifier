<div align="center">
    <img src="https://github.com/zehengl/ez-cuisine-classifier/raw/main/static/favicon.png" alt="logo" height="128">
</div>

# ez-cuisine-classifier

![coding_style](https://img.shields.io/badge/code%20style-black-000000.svg)

A Streamlit app to predict what is cooking

## Environment

- Python 3.9
- Windows 10

## Install

    python -m venv .venv
    .\.venv\Scripts\activate
    python -m pip install -U pip
    pip install -r requirements-dev.txt

If GPU/CUDA is available, add `--extra-index-url https://download.pytorch.org/whl/cu113` in `requirements-dev.txt` to install the CUDA-enabled version of `PyTorch`.

## Data Source

The training data is from kaggle's [Recipe Ingredients Dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset).

## Credits

- [Logo][1] by [Design Zone][2]

[1]: https://iconstore.co/icons/cafes-vector-icon-set/
[2]: https://iconstore.co/author/design-zone/
