"""Streamlit Demonstrator"""
# pylint: disable=C0413, E0401
import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from interface import config_interface
from interface.fragments.sidebar import custom_sidebar
from interface.pages.tab_cartoon_ranking import tab_cartoon_ranking
from interface.pages.tab_cartoon_ranking_results import tab_ranking_results_display

st.set_page_config(layout="wide", page_title="Streamlit Demonstrator", page_icon="🎨")

st.markdown("# Cartoonizer demonstrator")

custom_sidebar()


if st.session_state.get("page_mode") == config_interface.TAB_CARTOONS_RANKING:
    tab_cartoon_ranking()
elif st.session_state.get("page_mode") == config_interface.TAB_CARTOONS_RANKING_RESULTS:
    tab_ranking_results_display()
