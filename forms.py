from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired


class IngredientForm(FlaskForm):
    ingredients = StringField("Ingredients", validators=[DataRequired()])
    submit = SubmitField("Submit")
