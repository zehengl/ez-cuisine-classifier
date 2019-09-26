from joblib import load
from flask import Flask, render_template, request
from whitenoise import WhiteNoise

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")
model = load("model_sklearn")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ingredients = request.form["ingredients"]
        return render_template(
            "index.html",
            cuisine=model.predict([ingredients]).item(),
            ingredients=ingredients,
        )
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
