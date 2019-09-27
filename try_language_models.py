#%%
import json
import os
import random

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

train_json = json.load(open(os.path.join(data, "train.json")))

ingredients = [" ".join(item["ingredients"]) for item in train_json]
cuisine = [item["cuisine"] for item in train_json]

df = pd.DataFrame({"cuisine": cuisine, "ingredients": ingredients})

train_df, valid_df = train_test_split(
    df, stratify=df["cuisine"], test_size=0.2, random_state=1024
)

#%%
text_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=valid_df, path="")
lm_learner = language_model_learner(text_lm, arch=AWD_LSTM, drop_mult=0.2)

#%%
lm_learner.lr_find()
lm_learner.recorder.plot()

#%%
lm_learner.fit_one_cycle(1, 1e0)

#%%
lm_learner.lr_find()
lm_learner.recorder.plot()

#%%
lm_learner.unfreeze()
lm_learner.fit_one_cycle(1, (1e-2)/2)

#%%
lm_learner.save_encoder(model)

#%%
text_clas = TextClasDataBunch.from_df(
    train_df=train_df, valid_df=valid_df, vocab=text_lm.train_ds.vocab, path=""
)

#%%
clf = text_classifier_learner(text_clas, arch=AWD_LSTM, drop_mult=0.2)
clf.load_encoder(model)

#%%
clf.lr_find()
clf.recorder.plot()

#%%
clf.fit_one_cycle(1, 5e-1)

#%%
clf.lr_find()
clf.recorder.plot()

#%%
clf.unfreeze()
clf.fit_one_cycle(1, 2e-4)

#%%
lm_learner.predict("green bell")

#%%
clf.predict("wrap avacado beef")
