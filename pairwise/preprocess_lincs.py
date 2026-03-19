import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

def preprocess_lincs(file_path):

    return ''

def read_in_lincs(file_path, ids):
    subset = parse(file_path, cid=ids)
    return subset

if __name__ == "__main__":
    preprocess_lincs()