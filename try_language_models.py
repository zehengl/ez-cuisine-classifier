# %%
import json
import os

import pandas as pd
from fastai.text.all import (
    AWD_LSTM,
    Perplexity,
    TextDataLoaders,
    accuracy,
    language_model_learner,
    text_classifier_learner,
)

data = "data"

if __name__ == "__main__":

    train_json = json.load(open(os.path.join(data, "train.json")))

    ingredients = [" ".join(item["ingredients"]) for item in train_json]
    cuisine = [item["cuisine"] for item in train_json]

    df = pd.DataFrame({"cuisine": cuisine, "ingredients": ingredients})

    dls_lm = TextDataLoaders.from_df(
        df=df,
        text_col="ingredients",
        label_col="cuisine",
        is_lm=True,
        valid_pct=0.1,
    )

    learn = language_model_learner(
        dls_lm,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=[accuracy, Perplexity()],
    ).to_fp16()

    learn.lr_find()
    learn.fit_one_cycle(5, learn.recorder.lr)

    print(learn.predict("green bell", 1, temperature=0.75))

    learn.save_encoder("encoder")

    dls_clas = TextDataLoaders.from_df(
        df,
        text_vocab=dls_lm.vocab,
        text_col="ingredients",
        label_col="cuisine",
    )

    learn = text_classifier_learner(
        dls_clas,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=accuracy,
    )

    learn = learn.load_encoder("encoder")
    learn.lr_find()
    learn.fit_one_cycle(5, learn.recorder.lr)

    print(learn.predict("wrap avocado beef"))
