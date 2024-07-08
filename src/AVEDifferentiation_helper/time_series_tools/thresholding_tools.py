import numpy as np
import matplotlib.pyplot as plt
from .general_tools import compute_distance_xy

def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
 
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
 
    determinant = a1*b2 - a2*b1
 
    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX
        return [10**9, 10**9]
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return [x, y]
 
def compute_histogram(data, bins):
    """
    Compute the histogram for the given data.
    
    Parameters:
    data (numpy.ndarray): Input data (e.g., grayscale image values).
    bins (int): Number of bins for the histogram.
    
    Returns:
    hist (numpy.ndarray): Histogram counts.
    bin_edges (numpy.ndarray): Bin edges.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=(0, np.max(data)))
    return hist, bin_edges

def perpendicular_line_equation(slope, point_on_line, given_point):
    x1, y1 = point_on_line
    x0, y0 = given_point
    
    if slope == 0:
        slope_perpendicular = float('inf')
    else:
        slope_perpendicular = -1 / slope
    
    if slope_perpendicular == float('inf'):
        intercept_perpendicular = x0
    else:
        intercept_perpendicular = y0 - slope_perpendicular * x0
    
    return slope_perpendicular, intercept_perpendicular

def triangle_thresholding(hist):
    # Find the peak (maximum value) in the histogram
    peak_index = np.argmax(hist)
    peak_value = hist[peak_index]
    
    # Coordinates of the end point (right-most bin)
    end_index = len(hist) - 1
    end_value = hist[end_index]
    
    # Calculate the slope of the line from peak to end
    slope = (end_value - peak_value) / (end_index - peak_index)
    
    # Compute the line values
    line_values = peak_value + slope * (np.arange(peak_index, end_index + 1) - peak_index)
    
    # Initialize variables to find the maximum distance
    max_distance = -1
    threshold = peak_index
    
    for bin_idx in range(peak_index, end_index + 1):
        # Perpendicular distance at the threshold
        slope_perpendicular, intercept_perpendicular = perpendicular_line_equation(slope, [peak_index, peak_value], [bin_idx, hist[bin_idx]])
        perpendicular_value = intercept_perpendicular + slope_perpendicular * end_index

        A = [peak_index, peak_value]
        B = [end_index, end_value]
        C = [bin_idx, hist[bin_idx]]
        D = [end_index, perpendicular_value]
        
        intersection = lineLineIntersection(A, B, C, D)
        if intersection[0] == 10**9: continue
        distance = compute_distance_xy(bin_idx, intersection[0], hist[bin_idx], intersection[1])
        print(distance)
        if distance > max_distance:
            max_distance = distance
            threshold = bin_idx
    
    return threshold, peak_index, peak_value, slope

def plot_triangle_thresholding(hist, bin_idx, ax=None):

    # Find the peak (maximum value) in the histogram
    peak_index = np.argmax(hist)
    peak_value = hist[peak_index]
    
    # Coordinates of the end point (right-most bin)
    end_index = len(hist) - 1
    end_value = hist[end_index]
    
    # Calculate the slope of the line from peak to end
    slope = (end_value - peak_value) / (end_index - peak_index)
    
    # Compute the line values
    line_step = 0.1
    line_values = peak_value + slope * (np.arange(peak_index, end_index + line_step, line_step) - peak_index)
    
    # Perpendicular distance at the threshold
    slope_perpendicular, intercept_perpendicular = perpendicular_line_equation(slope, [peak_index, peak_value], [bin_idx, hist[bin_idx]])
    perpendicular_value = intercept_perpendicular + slope_perpendicular * end_index

    A = [peak_index, peak_value]
    B = [end_index, end_value]
    C = [bin_idx, hist[bin_idx]]
    D = [end_index, perpendicular_value]
    
    intersection = lineLineIntersection(A, B, C, D)

    # Plot histogram
    if ax is None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(hist)), hist, width=1, edgecolor='k', alpha=0.6, label='Histogram')
        
        # Plot the line from peak to end
        plt.plot(range(peak_index, end_index + 1), line_values, 'r--', lw=2, label='Line from Peak to End')
        
        # Plot the perpendicular line at the threshold
        plt.plot([bin_idx, intersection[0]], [hist[bin_idx], intersection[1]], 'g-', lw=2, label='Perpendicular Distance')
        
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.title('Triangle Thresholding')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()
    else:
                # Plot histogram
        ax.bar(range(len(hist)), hist, width=1, color=[0.7, 0.7, 0.0], edgecolor='k', alpha=0.6, label='Histogram')

        # Plot the line from peak to end
        ax.plot(range(peak_index, end_index + 1), line_values, 'r--', lw=2, label='Line from Peak to End')

        # Plot the perpendicular line at the threshold
        ax.plot([bin_idx, intersection[0]], [hist[bin_idx], intersection[1]], 'g-', lw=2, label='Perpendicular Distance')

        ax.set_xlabel('Bin')
        ax.set_ylabel('Frequency')
        ax.set_title('Triangle Thresholding')
