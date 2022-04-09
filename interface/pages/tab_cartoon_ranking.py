"""Tab Cartoon Ranking"""
# pylint: disable=C0413, E0401
import json
import os
import sys
from itertools import chain
from random import shuffle
from typing import Dict, List

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from interface.config_interface import HAND_EVALUATION_JSON


@st.experimental_singleton
def load_test_images_df() -> pd.DataFrame:
    """Load test images df"""
    return pd.read_csv(os.path.join("data", "pictures_test.csv"))


def filter_images_are_inresults_folder(df: pd.DataFrame) -> pd.DataFrame:
    """Filter images are inresults folder"""
    cartoons_results = get_cartoons_results_folders()
    all_cartoons = list(chain(*cartoons_results.values()))
    all_cartoons = [cartoon.split("\\")[-1] for cartoon in all_cartoons]
    df["is_in_results"] = df["name"].apply(lambda x: x in all_cartoons)

    return df[df["is_in_results"]].drop_duplicates(subset=["name"], keep="last")


@st.experimental_singleton
def get_cartoons_results_folders():
    """Lists all images in folders in the results folder"""
    folders_in_results = [
        folder
        for folder in os.listdir(os.path.join("data", "results"))
        if os.path.isdir(os.path.join("data", "results", folder))
    ]
    images_by_folder = {
        folder: [
            os.path.join("data", "results", folder, image)
            for image in os.listdir(os.path.join("data", "results", folder))
        ]
        for folder in folders_in_results
    }
    return images_by_folder


def update_element_id(offset: int, max_size: int):
    """Generate a function that changes the element_id."""

    def update_element_id_inner():
        """Function that update the element_id"""
        st.session_state["element_id"] = max(
            0, min(max_size - 1, st.session_state.get("element_id", 0) + offset)
        )

    return update_element_id_inner


def list_cartoons_from_image(image_path: str) -> List[str]:
    """List cartoonized images"""
    cartoons_results_folder = get_cartoons_results_folders()
    cartoons_from_image = []
    for folder in cartoons_results_folder:
        if image_path in [
            path.split("\\")[-1] for path in cartoons_results_folder[folder]
        ]:
            cartoons_from_image.append(
                os.path.join("data", "results", folder, image_path)
            )

    return cartoons_from_image


def tab_cartoon_ranking():
    """Tab cartoon ranking"""

    st.markdown("# Cartoon Ranking")

    df = filter_images_are_inresults_folder(load_test_images_df())

    if len(df) > 0:

        st.slider("Image considered", 0, len(df) - 1, 1, key="element_id")

        cols = st.columns(8)

        for i, col in zip([-100, -10, -1, 1, 10, 100], cols):
            with col:
                st.button(
                    str(i) if i < 0 else "+" + str(i),
                    on_click=update_element_id(i, len(df)),
                )

        st.markdown("## Original image")

        st.image(df.iloc[st.session_state["element_id"]]["path"])

        st.markdown("## Cartoonized images")

        image_name = df.iloc[st.session_state["element_id"]]["name"]

        cartoons_for_image = list_cartoons_from_image(image_name)

        shuffle(cartoons_for_image)

        if is_data_in_json(image_name):
            st.warning("Text was already ranked 👀!")

        with st.form(key="metrics_ranked"):

            kwargs = {
                "image_path": image_name,
                "models_list": [
                    cartoon_path.split("\\")[-2] for cartoon_path in cartoons_for_image
                ],
            }

            for i, cartoon_path in enumerate(cartoons_for_image):
                st.markdown(f"#### Model {i}")

                col1, col2, _ = st.columns([3, 8, 1])

                with col1:
                    st.number_input(
                        "Cartoon rank",
                        min_value=0,
                        max_value=len(cartoons_for_image),
                        value=i + 1,
                        key=cartoon_path.split("\\")[-2],
                    )

                with col2:

                    st.image(cartoon_path)

                with st.expander("Spoiler 👻"):
                    st.write("Model name is ", cartoon_path.split("\\")[-2])

            st.form_submit_button(
                label="Save metric ranking",
                help=None,
                on_click=save_metric_ranking,
                kwargs=kwargs,
            )
        if not st.session_state.get("is_ranking_ok"):
            st.warning("Ranking isn't correct... values weren't saved")
    else:
        st.write("No images in results folder 🥲")

        st.write(
            "Please read the documentation to learn how to generate the results folder"
        )


def save_metric_ranking(image_path: str, models_list: List[str]):
    """Save metric ranking"""
    if not os.path.exists(HAND_EVALUATION_JSON):
        # create the json file
        with open(HAND_EVALUATION_JSON, "w", encoding="utf-8") as json_file:
            json.dump({}, json_file, ensure_ascii=False, indent=4)
    ranking = {
        model_name: st.session_state.get(model_name) for model_name in models_list
    }
    if is_ranking_ok(ranking):
        with open(HAND_EVALUATION_JSON, "r+", encoding="utf-8") as json_file:
            data = json.load(json_file)
            data[hash(image_path)] = {"image_path": image_path, "ranking": ranking}
            json_file.seek(0)
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        st.session_state["element_id"] = st.session_state.get("element_id") + 1
        st.session_state["is_ranking_ok"] = True
    else:
        st.session_state["is_ranking_ok"] = False


def is_ranking_ok(ranking: Dict[str, int]) -> bool:
    """Checks that ranking is consistent"""
    values = ranking.values()
    # values should be unique
    if len(set(values)) != len(values):
        return False
    # between 1 and nb values should always be true with
    # constraints on input
    return True


def is_data_in_json(patient_text: str) -> bool:
    """Checks if data already in json"""
    with open(HAND_EVALUATION_JSON, encoding="utf-8") as eval_json:
        data = json.load(eval_json)

        if str(hash(patient_text)) in data:
            return True
        return False
