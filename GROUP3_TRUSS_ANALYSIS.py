import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




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






def plot_truss_structure(node_coords, element_nodes, loads_vector, title="Truss Structure"):
    plt.figure(figsize=(12, 8))
    
    # Convert coordinates for better scaling
    node_coords_km = node_coords / 1000 
    
    # Calculate plot boundaries
    x_coords = node_coords_km[:, 0]
    y_coords = node_coords_km[:, 1]
    x_center, y_center = np.mean(x_coords), np.mean(y_coords)
    x_range = max(x_coords) - min(x_coords) or 1
    y_range = max(y_coords) - min(y_coords) or 1
    
    #axis limits 
    plt.xlim([x_center - x_range*0.7, x_center + x_range*0.7])
    plt.ylim([y_center - y_range*0.7, y_center + y_range*0.7])

    #Plot elements
    for start, end in element_nodes:
        x = [node_coords_km[start][0], node_coords_km[end][0]]
        y = [node_coords_km[start][1], node_coords_km[end][1]]
        plt.plot(x, y, 'b-o', linewidth=1.5, markersize=4, markerfacecolor='blue')

    #force magnitude arrow scaling
    force_magnitudes = np.sqrt(loads_vector[::2]**2 + loads_vector[1::2]**2)
    max_force = np.max(force_magnitudes) or 1 
    
    # Auto-scale 
    arrow_scale = 0.1 * x_range / max_force 
    label_offset = x_range * 0.02  

    # Plot loads with proportional arrows
    for i in range(len(node_coords_km)):
        fx = loads_vector[2*i]
        fy = loads_vector[2*i+1]
        if fx != 0 or fy != 0:
            x, y = node_coords_km[i]
            dx = fx * arrow_scale
            dy = fy * arrow_scale
            
            plt.arrow(x, y, dx, dy, 
                      color='red', 
                      head_width=x_range*0.015, 
                      head_length=x_range*0.03,
                      length_includes_head=True,
                      width=x_range*0.003)
            
            # Calculate label position
            angle = np.arctan2(dy, dx)
            label_x = x + dx + label_offset * np.cos(angle)
            label_y = y + dy + label_offset * np.sin(angle)
            
            plt.text(label_x, label_y, 
                    f"{np.hypot(fx, fy)/1000:.1f} kN", 
                    color='darkred', fontsize=9, ha='center')

    plt.title(title)
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()



def plot_member_forces(node_coords, element_nodes, member_forces, title="Member Forces Analysis"):
    """Plot 2: Shows member forces with tension/compression visualization"""
    plt.figure(figsize=(12, 8))
    
    # Plot elements with force magnitude and type
    for i, (start, end) in enumerate(element_nodes):
        x = [node_coords[start][0], node_coords[end][0]]
        y = [node_coords[start][1], node_coords[end][1]]
        
        # Force-based styling
        force = member_forces[i]
        linewidth = 1 + abs(force) / 5000  # Scale line width with force
        color = 'blue' if force > 0 else 'red'  # Blue=Tension, Red=Compression
        
        plt.plot(x, y, color=color, linewidth=linewidth)
        
        # Add force label at midpoint
        mid_x = sum(x)/2
        mid_y = sum(y)/2
        plt.text(mid_x, mid_y, 
                f"{abs(force/1000):.1f} kN\n({'T' if force>0 else 'C'})", 
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()





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
    
    
    # In your main function after analysis:
    plot_truss_structure(node_coords, element_nodes, loads_vector)
    plot_member_forces(node_coords, element_nodes, member_forces)
    
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
