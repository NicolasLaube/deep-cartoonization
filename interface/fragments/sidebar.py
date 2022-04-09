"""Demonstrator sidebar"""
# pylint: disable=C0413, E0401
import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from interface.config_interface import TABS


def custom_sidebar():
    """Custom sidebar"""

    with st.sidebar:

        st.markdown("## Select tab:")

        st.radio("", TABS, index=0, key="page_mode")
