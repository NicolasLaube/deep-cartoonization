"""Config interface"""
import os

ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TAB_CARTOONS_RANKING = "Cartoons Ranking"
TAB_CARTOONS_RANKING_RESULTS = "Cartoons Ranking Results"

TABS = [TAB_CARTOONS_RANKING, TAB_CARTOONS_RANKING_RESULTS]

HAND_EVALUATION_JSON = os.path.join(ROOT_FOLDER, "interface", "hand_evaluation.json")
