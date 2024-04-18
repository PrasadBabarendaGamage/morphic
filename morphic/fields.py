import numpy as np

def generate_xi_on_line(line, num_points=4, dim=3, start_value=0, end_value=1):
    """Generate linearly spaced xi values along a local element line.

    Args:
      line: Line number to generate xi values for. These are defined
        sequentially following standard FEM numbering. E.g. for hexahedral
        elements in 3D space, there are 1-12 lines, which are numbered by
        considering all lines along xi1 (4 lines), followed by all lines along
        xi2 (4 lines), followed by all lines along xi3 (4 lines).
      num_points: Number of points generate along the xi direction.
      dim: Dimension of the element.
      start_value: Xi value to start generating xi values for.
      end_value: Xi value to finish generating xi values for.

    Returns:
      xi: Linearly spaced xi values along the specified line with the specified
        density.
    """

    if dim == 3:
        if line == 1:
            xi1 = np.linspace(start_value, end_value, num_points)
            xi2 = 0
            xi3 = 0
        elif line == 2:
            xi1 = np.linspace(start_value, end_value, num_points)
            xi2 = 1
            xi3 = 0
        elif line == 3:
            xi1 = np.linspace(start_value, end_value, num_points)
            xi2 = 0
            xi3 = 1
        elif line == 4:
            xi1 = np.linspace(start_value, end_value, num_points)
            xi2 = 1
            xi3 = 1

        elif line == 5:
            xi1 = 0
            xi2 = np.linspace(start_value, end_value, num_points)
            xi3 = 0
        elif line == 6:
            xi1 = 1
            xi2 = np.linspace(start_value, end_value, num_points)
            xi3 = 0
        elif line == 7:
            xi1 = 0
            xi2 = np.linspace(start_value, end_value, num_points)
            xi3 = 1
        elif line == 8:
            xi1 = 1
            xi2 = np.linspace(start_value, end_value, num_points)
            xi3 = 1

        elif line == 9:
            xi1 = 0
            xi2 = 0
            xi3 = np.linspace(start_value, end_value, num_points)
        elif line == 10:
            xi1 = 1
            xi2 = 0
            xi3 = np.linspace(start_value, end_value, num_points)
        elif line == 11:
            xi1 = 0
            xi2 = 1
            xi3 = np.linspace(start_value, end_value, num_points)
        elif line == 12:
            xi1 = 1
            xi2 = 1
            xi3 = np.linspace(start_value, end_value, num_points)

        else:
            raise ValueError(
                'Local element number between 1-12 are supported for '
                'hexahedral elements.')

        X, Y, Z = np.meshgrid(xi1, xi2, xi3)
        xi = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size)),
            Z.reshape((Z.size))]).T

    elif dim == 2:
        if line == 1:
            xi1 = np.linspace(start_value, end_value, num_points)
            xi2 = 0
        elif line == 2:
            xi1 = np.linspace(start_value, end_value, num_points)
            xi2 = 1

        elif line == 3:
            xi1 = 0
            xi2 = np.linspace(start_value, end_value, num_points)
        elif line == 4:
            xi1 = 1
            xi2 = np.linspace(start_value, end_value, num_points)

        X, Y = np.meshgrid(xi1, xi2)
        xi = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T

    return xi


def generate_xi_on_face(face, value, num_points=[4, 4], dim=3):
    """
    Generate a grid of points within each element

    Keyword arguments:
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions
    """
    num_points = np.atleast_1d(num_points)
    if len(num_points)==1:
        num_points = np.array([num_points[0], num_points[0]])

    if dim == 3:
        if face == "xi1":
            xi1 = [value]
            xi2 = np.linspace(0., 1., num_points[0])
            xi3 = np.linspace(0., 1., num_points[1])
        elif face == "xi2":
            xi1 = np.linspace(0., 1., num_points[0])
            xi2 = [value]
            xi3 = np.linspace(0., 1., num_points[1])
        elif face == "xi3":
            xi1 = np.linspace(0., 1., num_points[0])
            xi2 = np.linspace(0., 1., num_points[1])
            xi3 = [value]
        X, Y, Z = np.meshgrid(xi1, xi2, xi3)
        xi = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size)),
            Z.reshape((Z.size))]).T
    elif dim == 2:
        xi1 = np.linspace(0., 1., num_points[0])
        xi2 = np.linspace(0., 1., num_points[1])
        X, Y = np.meshgrid(xi1, xi2)
        xi = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T

    return xi


def generate_xi_grid_fem(num_points=[4, 4, 4], dim=3):
    # Generate a grid of points within each element
    xi1 = np.linspace(0., 1., num_points[0])
    xi2 = np.linspace(0., 1., num_points[1])
    if dim == 2:
        X, Y = np.meshgrid(xi1, xi2)
        XiNd = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    else:
        xi3 = np.linspace(0., 1., num_points[2])
        X, Y, Z = np.meshgrid(xi1, xi2, xi3)
        XiNd = np.array([
            Z.reshape((Z.size)),
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    return XiNd