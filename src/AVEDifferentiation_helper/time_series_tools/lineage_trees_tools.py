import numpy as np
from embdevtools.celltrack.core.plot.napari_tools import get_lineage_graph, napari_tracks, get_lineage_root, get_whole_lineage, get_all_daughters, get_lineage_ends, get_daughters
def assign_positions(graph, lab, level, dxs, mother_position, positions):
    daughters = get_daughters(graph, lab)
    first = True
    for daughter in daughters:
        if first:
            dx = -dxs[level]/2
        else: 
            dx = dxs[level]/2
        
        daughter_position = mother_position+dx
        positions[daughter] = daughter_position
        first=False
        assign_positions(graph, daughter, level+1, dxs, daughter_position, positions)

def get_longest_route(graph, ends):
    maxn=0
    for end in ends:
        n = 0
        lab = end
        try:
            while True:
                new_lab = graph[lab]
                lab = new_lab
                n+=1
        except KeyError:
            maxn = np.maximum(n, maxn)
    return maxn

# Step 5: Function to plot a vertical line with a color gradient
def plot_vertical_gradient_line(ax, x, cmap, norm, cell, values, linewidth=5):
    num_segments = len(values)
    y_values = cell.times
    for i in range(num_segments - 1):
        value_start = values[i]
        value_end = values[i+1]
        color = cmap(norm(value_start + (value_end - value_start) * (i / (num_segments - 1))))
        _ = ax.plot([x, x], [y_values[i], y_values[i + 1]], color=color, linewidth=linewidth, zorder=1)