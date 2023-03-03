import streamlit as st
from joblib import load


@st.cache(allow_output_mutation=True)
def load_model():
    model = load("model_sklearn")
    return model


st.set_page_config(page_title="ez-cuisine-classifier", page_icon=":fork_and_knife:")
_, center, _ = st.columns([2, 1, 2])
with center:
    st.image(
        "https://github.com/zehengl/ez-cuisine-classifier/raw/main/static/favicon.png",
        use_column_width=True,
    )
st.title("ez-cuisine-classifier")
st.caption("A Streamlit app to predict what is cooking")

model = load_model()
ingredients = st.text_input("Ingredients")

if ingredients:
    st.caption("Smells Good!")
    st.markdown(f"Are you probably cooking **{model.predict([ingredients]).item()}**?")
