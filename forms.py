from wtforms import (SelectField, TextAreaField, SubmitField, validators)
from flask_wtf import FlaskForm


class CryptoForm(FlaskForm):
    crypto_select = SelectField('Choose a crypto...')
    submit = SubmitField('Get Price Data', render_kw={'class': 'button-form'})

    def __init__(self, crypto_options=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_options = crypto_options or []
        self.crypto_select.choices = self.rag_options


class CryptoFormAI(FlaskForm):
    query = TextAreaField(
        "",
        validators=[validators.InputRequired("A query is required.")],
        render_kw={"placeholder": "Ask about crypto"})
    submit = SubmitField("Get Answer")
