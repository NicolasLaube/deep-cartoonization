"""Config interface"""
import os

ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TAB_CARTOONS_RANKING = "Cartoons Ranking"

TABS = [
    TAB_CARTOONS_RANKING,
]

HAND_EVALUATION_JSON = os.path.join(ROOT_FOLDER, "interface", "hand_evaluation.json")
