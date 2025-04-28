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
    data = read_single_sheet("truss_analysis.xlsx")

    nodes = data['NODES']
    elements = data['ELEMENTS']
    loads = data['LOADS']
    supports = data['SUPPORTS']

    node_coords = nodes[['X', 'Y']].values
    num_nodes = len(node_coords)
    
    element_nodes = elements[['StartNode', 'EndNode']].values.astype(int) - 1
    element_areas = elements['Area'].astype(float).values
    element_E = elements['E'].astype(float).values
    num_elements = len(element_nodes)

    loads_vector = np.zeros(2 * num_nodes)
    for _, row in loads.iterrows():
        node = int(row['Node']) - 1
        loads_vector[2 * node] = row['Fx']
        loads_vector[2 * node + 1] = row['Fy']

    support_conditions = np.zeros(2 * num_nodes, dtype=int)
    for _, row in supports.iterrows():
        node = int(row['Node']) - 1
        support_conditions[2 * node] = row['Xfixed']
        support_conditions[2 * node + 1] = row['Yfixed']

    global_stiffness = np.zeros((2 * num_nodes, 2 * num_nodes))

    for i in range(num_elements):
        start, end = element_nodes[i]
        E = element_E[i]
        A = element_areas[i]

        x1, y1 = node_coords[start]
        x2, y2 = node_coords[end]
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)

        c = dx / L
        s = dy / L

        k = (E * A / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

        idx = [2 * start, 2 * start + 1, 2 * end, 2 * end + 1]

        for r in range(4):
            for c_ in range(4):
                global_stiffness[idx[r], idx[c_]] += k[r, c_]

    for i in range(2 * num_nodes):
        if support_conditions[i] == 1:
            global_stiffness[i, :] = 0
            global_stiffness[:, i] = 0
            global_stiffness[i, i] = 1
            loads_vector[i] = 0

    try:
        displacements = np.linalg.solve(global_stiffness, loads_vector)
    except np.linalg.LinAlgError:
        print("Error: The structure is unstable or improperly constrained!")
        return

    member_forces = np.zeros(num_elements)
    member_stresses = np.zeros(num_elements)
    for i in range(num_elements):
        start, end = element_nodes[i]
        E = element_E[i]
        A = element_areas[i]

        u1, v1 = displacements[2 * start], displacements[2 * start + 1]
        u2, v2 = displacements[2 * end], displacements[2 * end + 1]

        x1, y1 = node_coords[start]
        x2, y2 = node_coords[end]
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)

        c = dx / L
        s = dy / L

        delta = (u2 - u1) * c + (v2 - v1) * s
        force = (E * A / L) * delta
        member_forces[i] = force
        member_stresses[i] = force / A
        
        # Displacement 
    disp_df = pd.DataFrame([{
        'Node': i + 1,
        'X-Displacement (m)': displacements[2 * i],
        'Y-Displacement (m)': displacements[2 * i + 1]
    } for i in range(num_nodes)])

    # Forces and Stress
    force_df = pd.DataFrame([{
        'Element': i + 1,
        'Force (N)': member_forces[i],
        'Stress (Pa)': member_stresses[i],
        'Nature': "Tension" if member_forces[i] > 0 else "Compression"
    } for i in range(num_elements)])
    
    with pd.ExcelWriter('truss_analysis.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
     disp_df.to_excel(writer, sheet_name='Displacements', index=False)
     force_df.to_excel(writer, sheet_name='Member Forces', index=False)
     
    try:
        allowable_stress = float(input("\nEnter the allowable stress (Pa): "))    #0.6Fy
        
        if (force_df['Stress (Pa)'].abs() > allowable_stress).any():
            print("Optimization Required")
        else:
            print("No Optimization Needed")
    except ValueError:
        print("Invalid input for allowable stress. Please enter a numeric value.")

if __name__ == "__main__":
    main()
