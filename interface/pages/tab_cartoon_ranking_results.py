"""Tab cartoon ranking resutls"""
# pylint: disable=C0413, E0401
import json
import os
import sys
from collections import defaultdict

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from interface.config_interface import HAND_EVALUATION_JSON


def tab_ranking_results_display():
    """Tab ranking display results"""

    st.markdown("## Cartoon Ranking Results")

    st.dataframe(summarize_hand_ranks())


def load_data():
    """Load data"""
    if not os.path.exists(HAND_EVALUATION_JSON):
        # create the json file
        with open(HAND_EVALUATION_JSON, "w", encoding="utf-8") as json_file:
            json.dump({}, json_file, ensure_ascii=False, indent=4)
    with open(HAND_EVALUATION_JSON, "r", encoding="utf-8") as json_file:
        hand_evaluation = json.load(json_file)
    return hand_evaluation


def summarize_hand_ranks() -> pd.DataFrame:
    """Summarize hand ranks"""

    data = load_data()
    scores = defaultdict(lambda: defaultdict(int))  # type: ignore
    method_count = defaultdict(int)  # type: ignore
    for cartoon_data in data:
        for method in data[cartoon_data]["ranking"]:
            rank = data[cartoon_data]["ranking"][method]
            scores[method][f"rank_{rank}"] += 1
            method_count[method] += 1
    for method in scores:
        for rank in scores[method]:
            if method_count[method] != 0:
                scores[method][rank] = (
                    float(scores[method][rank]) / method_count[method]
                ) * 100
            else:
                scores[method][rank] = 0

    scores_df = pd.DataFrame(scores).fillna(0).sort_index(axis=1)
    scores_df = scores_df.astype(int).sort_index(axis=0)
    for column in scores_df.columns:
        scores_df[column] = scores_df[column].astype(str) + "%"

    return scores_df.style.highlight_max(axis=1, color="lightgreen")
