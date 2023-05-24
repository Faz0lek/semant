"""Constants and utility functions for NSP

Date -- 24.05.2023
Author -- Martin Kostelnik
"""

CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"
MODELS_PATH = r"../../models"


def load_data(path: str):
    sentences = [["Já jsem Martin a včera jsem šel do obchodu.", "Venku bylo hezké počasí.", 0],
                 ["Umělá inteligence se stále zlepšuje.", "Nikdy však nebude dost chytrá.", 0],
                 ["Strážci galaxie byl skvělý film.", "Není ale lepší než Pán prstenů.", 0],
                 ["Brno je nejlepší město v České republice", "Na cestě stojí auto.", 1]]

    return sentences
