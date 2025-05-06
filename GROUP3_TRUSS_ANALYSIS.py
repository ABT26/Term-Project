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
    plt.show(block=False)
    plt.pause(2)




    


def plot_member_forces(node_coords, element_nodes, member_forces, title="Member Forces Analysis"):
    """Plot 2: Shows member forces with tension/compression visualization"""
    plt.figure(figsize=(12, 8))
    
    # Set thickness parameters
    min_thickness = 0.5  # Thinnest line for smallest force
    max_thickness = 10   # Thickest line for largest force
    
    # Filter out zero forces if present to avoid division issues
    non_zero_forces = [f for f in member_forces if abs(f) > 1e-6]
    
    if not non_zero_forces:  # If all forces are zero
        non_zero_forces = [0]  # Fallback to prevent errors
    
    min_force = min(abs(f) for f in non_zero_forces)
    max_force = max(abs(f) for f in non_zero_forces)
    
    # Plot elements with force magnitude and type
    for i, (start, end) in enumerate(element_nodes):
        x = [node_coords[start][0], node_coords[end][0]]
        y = [node_coords[start][1], node_coords[end][1]]
        
        force = member_forces[i]
        abs_force = abs(force)
        
        # Normalize force magnitude between min and max thickness
        if max_force == min_force:  # All forces are equal
            thickness = (max_thickness + min_thickness)/2
        else:
            # Scale thickness proportionally between min and max
            normalized = (abs_force - min_force) / (max_force - min_force)
            thickness = min_thickness + normalized * (max_thickness - min_thickness)
        
        color = 'blue' if force > 0 else 'red'  # Blue=Tension, Red=Compression
        
        plt.plot(x, y, color=color, linewidth=thickness)
        
        # Add force label at midpoint
        mid_x = sum(x)/2
        mid_y = sum(y)/2
        
        plt.text(mid_x, mid_y, 
            f"M{i+1}\n{abs_force/1000:.1f} kN\n({'T' if force>0 else 'C'})", 
            ha='center', va='center', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show(block=False)
    plt.pause(1)



def plot_displaced_structure(node_coords, element_nodes, displacements, title="Displaced Structure"):
    plt.figure(figsize=(12, 8))
    
    # Convert original coordinates to kilometers
    node_coords_km = node_coords / 1000
    
    # Calculate automatic scaling factor for displacements
    disp_magnitudes = np.sqrt(displacements[::2]**2 + displacements[1::2]**2)
    max_disp = np.max(disp_magnitudes) if len(disp_magnitudes) > 0 else 0
    
    if max_disp > 0:
        # Calculate structure dimensions in meters
        x_coords = node_coords[:, 0]
        y_coords = node_coords[:, 1]
        x_range = np.ptp(x_coords)  # Peak-to-peak (max - min)
        y_range = np.ptp(y_coords)
        structure_size = max(x_range, y_range)
        
        # Scale factor to make max displacement 10% of structure size
        scale_factor = (0.05 * structure_size) / max_disp
    else:
        scale_factor = 1  # No displacement
    
    # Calculate displaced coordinates (in meters) and convert to km
    displaced_coords = node_coords + displacements.reshape(-1, 2) * scale_factor
    displaced_coords_m = displaced_coords / 1000
    
    # Combine coordinates for axis limits
    all_x = np.concatenate([node_coords_km[:, 0], displaced_coords_m[:, 0]])
    all_y = np.concatenate([node_coords_km[:, 1], displaced_coords_m[:, 1]])
    
    # Set plot boundaries
    x_center, y_center = np.mean(all_x), np.mean(all_y)
    x_range_plot = np.ptp(all_x) or 1
    y_range_plot = np.ptp(all_y) or 1
    
    plt.xlim([x_center - x_range_plot*0.7, x_center + x_range_plot*0.7])
    plt.ylim([y_center - y_range_plot*0.7, y_center + y_range_plot*0.7])
    
    # Plot original structure (dashed lines)
    for start, end in element_nodes:
        x = [node_coords_km[start][0], node_coords_km[end][0]]
        y = [node_coords_km[start][1], node_coords_km[end][1]]
        plt.plot(x, y, 'b--', linewidth=1, alpha=0.5)
    
    # Plot displaced structure (solid red lines)
    for start, end in element_nodes:
        x = [displaced_coords_m[start][0], displaced_coords_m[end][0]]
        y = [displaced_coords_m[start][1], displaced_coords_m[end][1]]
        plt.plot(x, y, 'r-', linewidth=1.5)
    
    # Create legend proxies
    plt.plot([], [], 'b--', label='Original Structure')
    plt.plot([], [], 'r-', label=f'Deformed Structure')
    plt.title(f"{title}\nDisplacements exaggerated for visualization")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
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
    
    
   
    
    
    try:
        # Get allowable stress from user
        allowable_stress = float(input("\nEnter the allowable stress (Pa): "))
        
        # Add 'Optimization Needed' column to force_df
        force_df['Optimization Needed'] = np.where(
            force_df['Stress (Pa)'].abs() > allowable_stress, 
            'Required', 
            'Not Required'
        )
        
        # Write updated results to Excel (replaces existing sheets)
        with pd.ExcelWriter('truss_analysis.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            disp_df.to_excel(writer, sheet_name='Displacements', index=False)
            force_df.to_excel(writer, sheet_name='Member Forces', index=False)
        
       
        
        # Print members requiring optimization
        if (force_df['Optimization Needed'] == 'Required').any():
            required_elements = force_df[force_df['Optimization Needed'] == 'Required']['Element'].tolist()
            print(f"Optimization Required for Members: {', '.join(map(str, required_elements))}")
        else:
            print("No Optimization Needed.")
            
        # # In your main function after analysis:
        plot_truss_structure(node_coords, element_nodes, loads_vector)
        plot_member_forces(node_coords, element_nodes, member_forces) 
        plot_displaced_structure(node_coords, element_nodes, displacements)   
    except ValueError:
        print("Invalid input. Please enter a numeric value for allowable stress.")
if __name__ == "__main__":
    main()
