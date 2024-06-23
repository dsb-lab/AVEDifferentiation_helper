import numpy as np

# Define useful functions
def compute_distance_xy(x1, x2, y1, y2):
    """
    Parameters
    ----------
    x1 : number
        x coordinate of point 1
    x2 : number
        x coordinate of point 2
    y1 : number
        y coordinate of point 1
    y2 : number
        y coordinate of point 2

    Returns
    -------
    dist : number
        euclidean distance between points (x1, y1) and (x2, y2)
    """
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def divide_line_1d(x1, x2, n):
    """
    Divide a 1D line segment from x1 to x2 into n equal parts.
    Coordinates of the division points are rounded to the nearest integer.
    """
    points = np.linspace(x1, x2, n + 1)
    int_points = np.round(points).astype(int)
    return int_points
