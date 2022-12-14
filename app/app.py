import streamlit as st
from tensorflow import keras
from PIL import Image
import io

import model


def configure():
    st.set_page_config(page_title="Low-light image enhancement")
    if "model" not in st.session_state:
        st.session_state["model"]: keras.Model = model.create_model()


def describe_service():
    st.title("Low-light image enhancement")
    st.subheader("Just upload your low-light image and get the processed one!")


@st.experimental_memo
def call_model(uploaded_file):
    return model.run_model(uploaded_file, st.session_state["model"])


def process_image():
    uploaded_file: io.BytesIO = st.file_uploader(
        label="Choose a file (you can upload new files without refreshing the page)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file:
        placeholder: st.delta_generator.DeltaGenerator = st.empty()
        placeholder.info("The image is being processed. It may take some time. Wait, please...")
        image: Image.Image = call_model(uploaded_file)
        placeholder.empty()
        placeholder.image(image)
        with open("user_data/output.jpg", "rb") as file:
            st.download_button(label="Download lightened image", data=file, file_name="lightened.jpg", mime="image/jpg")


def main():
    describe_service()
    process_image()


if __name__ == "__main__":
    configure()
    main()
