from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField
from wtforms.validators import DataRequired


class IngredientForm(FlaskForm):
    ingredients = TextField("Ingredients", validators=[DataRequired()])
    submit = SubmitField("Submit")
