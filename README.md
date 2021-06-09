<div align="center">
    <img src="https://github.com/zehengl/ez-cuisine-classifier/raw/main/static/favicon.png" alt="logo" height="196">
</div>

# ez-cuisine-classifier

![coding_style](https://img.shields.io/badge/code%20style-black-000000.svg)

A Python application to predict what is cooking

## Environment

- Python 3.7
- Windows 10

## Install

    python -m venv venv
    .\venv\Scripts\activate
    python -m pip install -U pip setuptools
    pip install -r requirements.txt

Use `pip install -r requirements-dev.txt` for development.
It will install `pylint` and `black` to enable linting and auto-formatting.
It also installs `jupyter` to allow notebook experience in VS Code when creating machine learning models.

## Data Source

The training data is from kaggle's [Recipe Ingredients Dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset).

## Demo

![watch](demo.gif)

<hr>

<sup>

## Credits

- [Logo][1] by [Design Zone][2]

</sup>

[1]: https://iconstore.co/icons/cafes-vector-icon-set/
[2]: https://iconstore.co/author/design-zone/
