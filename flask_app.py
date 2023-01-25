import os

from joblib import load
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from whitenoise import WhiteNoise

from forms import IngredientForm


app = Flask(__name__)
Bootstrap(app)
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "SECRET_KEY")
app.config["BOOTSTRAP_SERVE_LOCAL"] = True
model = load("model_sklearn")


def get_cuisine(ingredients):
    if not ingredients:
        return None

    return model.predict([ingredients]).item()


@app.route("/", methods=["get", "post"])
def index():
    form = IngredientForm(request.form)
    cuisine = get_cuisine(form.ingredients.data)

    return render_template("index.html", form=form, cuisine=cuisine)


if __name__ == "__main__":
    app.run(debug=True)
