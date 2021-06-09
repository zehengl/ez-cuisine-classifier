import json
import os

import pandas as pd
from fastai.text import (
    AWD_LSTM,
    TextClasDataBunch,
    TextLMDataBunch,
    language_model_learner,
    text_classifier_learner,
)
from sklearn.model_selection import train_test_split


data = "data"
model = "model_fastai"

if __name__ == "__main__":

    train_json = json.load(open(os.path.join(data, "train.json")))

    ingredients = [" ".join(item["ingredients"]) for item in train_json]
    cuisine = [item["cuisine"] for item in train_json]

    df = pd.DataFrame({"cuisine": cuisine, "ingredients": ingredients})

    train_df, valid_df = train_test_split(
        df,
        stratify=df["cuisine"],
        test_size=0.2,
        random_state=1024,
    )

    text_lm = TextLMDataBunch.from_df(
        train_df=train_df,
        valid_df=valid_df,
        path="",
    )
    lm_learner = language_model_learner(
        text_lm,
        arch=AWD_LSTM,
        drop_mult=0.2,
    )

    lm_learner.lr_find()
    lm_learner.recorder.plot(suggestion=True)

    lm_learner.fit_one_cycle(1, lm_learner.recorder.min_grad_lr)

    lm_learner.save_encoder(model)

    text_clas = TextClasDataBunch.from_df(
        train_df=train_df,
        valid_df=valid_df,
        vocab=text_lm.train_ds.vocab,
        path="",
    )

    clf = text_classifier_learner(
        text_clas,
        arch=AWD_LSTM,
        drop_mult=0.2,
    )
    clf.load_encoder(model)

    clf.lr_find()
    clf.recorder.plot(suggestion=True)

    clf.fit_one_cycle(1, clf.recorder.min_grad_lr)

    print(lm_learner.predict("green bell"))

    print(text_clas.train_ds.y.c2i)

    print(clf.predict("wrap avocado beef"))
