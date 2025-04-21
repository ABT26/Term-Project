import numpy as np
import pandas as pd

def read_single_sheet(filepath):
    df = pd.read_excel(filepath, header=None)

    sections = {}
    struct_input = None
    data_values = []

    for _, row in df.iterrows():
        if pd.isna(row[0]):
            continue
        elif isinstance(row[0], str) and row[0].strip().isupper():
            if struct_input and data_values:
                sections[struct_input] = pd.DataFrame(
                    data_values[1:], columns=data_values[0]
                )
            struct_input = row[0].strip()
            data_values = []
        else:
            data_values.append(row.tolist())

    if struct_input and data_values:
        sections[struct_input] = pd.DataFrame(
            data_values[1:], columns=data_values[0]
        )

    return sections

def main():
    data = read_single_sheet("truss_dataset.xlsx")

    nodes = data['NODES']
    elements = data['ELEMENTS']
    loads = data['LOADS']
    supports = data['SUPPORTS']

    node_coords = nodes[['X', 'Y']].values
    num_nodes = len(node_coords)