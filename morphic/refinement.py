import numpy as np
from morphic.mesher import Mesh

def get_num_elem_nodes(basis):
    """
    Returns the number of element nodes based on the given basis.

    Parameters:
    basis (list): A list of basis functions.

    Returns:
    list: A list of integers representing the number of element nodes for each dimension.

    Raises:
    ValueError: If the basis is not supported.

    """
    if basis == ['L1', 'L1', 'L1']:
        num_elem_nodes = [2, 2, 2]
    elif basis == ['L2', 'L2', 'L2']:
        num_elem_nodes = [3, 3, 3]
    elif basis ==  ['L3', 'L3', 'L3']:
        num_elem_nodes = [4, 4, 4] 
    else:  
        raise ValueError('Only 3D Lagrange elements up to the 3rd order (Cubic) are supported.')

    return num_elem_nodes

import numpy as np

def generate_xi_grid(num_points=[4, 4, 4], dim=3):
    """
    Generate a grid of points within each element.

    Parameters:
    - num_points (list): A list of integers representing the number of points in each dimension of the grid. Default is cubic lagrange, i.e. [4, 4, 4].
    - dim (int): The dimension of the grid. Default is 3.

    Returns:
    - XiNd (ndarray): An array of shape (N, dim), where N is the total number of points in the grid and dim is the dimension of the grid.
    """

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

def create_morphic_mesh(basis, xn, xe): 
    """
    Create a morphic mesh using the given basis, node values, and element values.

    Parameters:
    - basis (str): The basis for the mesh.
    - xn (list): The node values.
    - xe (list): The element values.

    Returns:
    - mesh (Mesh): The generated morphic mesh.
    """

    mesh = Mesh()

    for node_id, value in enumerate(xn):
        mesh.add_stdnode(node_id + 1, value, group='_default')

    for element_id, element in enumerate(xe):
        mesh.add_element(element_id + 1, basis, xe[element_id] + 1)

    mesh.generate(True)

    return mesh

def mesh_refine_along_xi1(mesh, basis=["L3", "L3", "L3"], fac=2): 
    """
    Refines a mesh along the xi1 direction.

    Args:
        mesh (MorphicMesh): The input mesh to be refined.
        basis (list, optional): The basis functions to be used for refinement. Defaults to ["L3", "L3", "L3"].
        fac (int, optional): The refinement factor. Defaults to 2.

    Returns:
        MorphicMesh: The refined mesh.

    Raises:
        None

    Examples:
        # Create a mesh 
        xn, xe = generate_element_nodes(["L3", "L3", "L3"], [2, 2, 2], extent=[1, 1, 1])
        mesh = create_morphic_mesh(["L3", "L3", "L3"], xn, xe)

        # Refine the mesh along the xi1 direction
        refined_mesh = mesh_refine_along_xi1(mesh, basis=["L3", "L3", "L3"], fac=2)
    """
    
    num_elem_nodes = get_num_elem_nodes(basis)
    xi_grid = generate_xi_grid(num_elem_nodes, dim=3)
    elem_partitions = [(n / fac, (n + 1) / fac) for n in range(fac)]

    xi_partitions = [np.array([xi_grid[:, 0] * (b[1] - b[0]) + b[0],
                               xi_grid[:, 1],
                               xi_grid[:, 2]]).T for b in elem_partitions]

    mapping = {}
    refined_xe = []
    for elem_idx, elem in enumerate(mesh.elements):
        for xi_part in xi_partitions:
            new_elem_nodes = elem.evaluate(xi_part)
            elem_node_ids = []
            for node in new_elem_nodes:
                if tuple(node) not in mapping:
                    mapping[tuple(node)] = len(mapping)
                elem_node_ids.append(mapping[tuple(node)])
            refined_xe.append(elem_node_ids)

    refined_xn_list = [list(point) for point in mapping.keys()]
    refined_xn = np.array(refined_xn_list) 
    refined_xe = np.array(refined_xe)

    refined_mesh = create_morphic_mesh(basis, refined_xn, refined_xe)

    return refined_mesh, refined_xn, refined_xe


def mesh_refine_along_xi2(mesh, basis=["L3", "L3", "L3"], fac=2): 
    """
    Refines a mesh along the xi2 direction.

    Args:
        mesh (MorphicMesh): The input mesh to be refined.
        basis (list, optional): The basis functions to be used for refinement. Defaults to ["L3", "L3", "L3"].
        fac (int, optional): The refinement factor. Defaults to 2.

    Returns:
        MorphicMesh: The refined mesh.

    Raises:
        None

    Examples:
        # Create a mesh 
        xn, xe = generate_element_nodes(["L3", "L3", "L3"], [2, 2, 2], extent=[1, 1, 1])
        mesh = create_morphic_mesh(["L3", "L3", "L3"], xn, xe)

        # Refine the mesh along the xi1 direction
        refined_mesh = mesh_refine_along_xi1(mesh, basis=["L3", "L3", "L3"], fac=2)
    """

    num_elem_nodes = get_num_elem_nodes(basis)
    xi_grid = generate_xi_grid(num_elem_nodes, dim=3)

    elem_partitions = [(n / fac, (n + 1) / fac) for n in range(fac)]

    xi_partitions = [np.array([xi_grid[:, 0],
                               xi_grid[:, 1] * (b[1] - b[0]) + b[0],
                               xi_grid[:, 2]]).T for b in elem_partitions]

    mapping = {}
    refined_xe = []
    for elem_idx, elem in enumerate(mesh.elements):
        for xi_part in xi_partitions:
            new_elem_nodes = elem.evaluate(xi_part)
            elem_node_ids = []
            for node in new_elem_nodes:
                if tuple(node) not in mapping:
                    mapping[tuple(node)] = len(mapping)
                elem_node_ids.append(mapping[tuple(node)])
            refined_xe.append(elem_node_ids)

    refined_xn_list = [list(point) for point in mapping.keys()]
    refined_xn = np.array(refined_xn_list) 
    refined_xe = np.array(refined_xe) 

    refined_mesh = create_morphic_mesh(basis, refined_xn, refined_xe)

    return refined_mesh, refined_xn, refined_xe

def mesh_refine_along_xi3(mesh, basis=["L3", "L3", "L3"], fac=2):
    """
    Refines a mesh along the xi3 direction.

    Args:
        mesh (MorphicMesh): The input mesh to be refined.
        basis (list, optional): The basis functions to be used for refinement. Defaults to ["L3", "L3", "L3"].
        fac (int, optional): The refinement factor. Defaults to 2.

    Returns:
        MorphicMesh: The refined mesh.

    Raises:
        None

    Examples:
        # Create a mesh 
        xn, xe = generate_element_nodes(["L3", "L3", "L3"], [2, 2, 2], extent=[1, 1, 1])
        mesh = create_morphic_mesh(["L3", "L3", "L3"], xn, xe)

        # Refine the mesh along the xi1 direction
        refined_mesh = mesh_refine_along_xi1(mesh, basis=["L3", "L3", "L3"], fac=2)
    """
    num_elem_nodes = get_num_elem_nodes(basis)
    xi_grid = generate_xi_grid(num_elem_nodes, dim=3)
    elem_partitions = [(n / fac, (n + 1) / fac) for n in range(fac)]  # e.g. for fac=2, [(0, 0.5), (0.5, 1)]

    xi_partitions = [np.array([xi_grid[:, 0],
                               xi_grid[:, 1],
                               xi_grid[:, 2] * (b[1] - b[0]) + b[0]]).T for b in elem_partitions]

    mapping = {}
    refined_xe = []
    for elem_idx, elem in enumerate(mesh.elements):
        for xi_part in xi_partitions:
            new_elem_nodes = elem.evaluate(xi_part)
            elem_node_ids = []
            for node in new_elem_nodes:
                # Check if point is already in the mapping, if not add it
                if tuple(node) not in mapping:
                    mapping[tuple(node)] = len(mapping)
                elem_node_ids.append(mapping[tuple(node)])
            refined_xe.append(elem_node_ids)

    # Generate the list of refined mesh nodes based on the mapping
    refined_xn_list = [list(point) for point in mapping.keys()]
    refined_xn = np.array(refined_xn_list) 
    refined_xe = np.array(refined_xe)

    # Update the refined mesh with the new nodes 
    refined_mesh = create_morphic_mesh(basis, refined_xn, refined_xe)
    

    return refined_mesh, refined_xn, refined_xe


def full_refinement(mesh, basis=["L3", "L3", "L3"], facs=[2, 2, 2]):
    """
    Perform full refinement of a mesh.

    Args:
        mesh (MorphicMesh): The input mesh to be refined.
        basis (list, optional): The basis functions to be used for refinement. Defaults to ["L3", "L3", "L3"].
        facs (list, optional): The refinement factors for each xi direction. Defaults to [2, 2, 2].

    Returns:
        MorphicMesh: The refined mesh. 

    Examples:
        # Create a mesh 
        xn, xe = morphic.generation.generate_element_nodes(["L3", "L3", "L3"], [2, 2, 2], extent=[1, 1, 1])
        mesh = morphic.Mesh()

        # Add nodes.
        for node_id, value in enumerate(xn):
            mesh.add_stdnode(node_id + 1, value, group='_default')

        # Add elements.
        for element_id, element in enumerate(xe):
            mesh.add_element(element_id + 1, ['L3', 'L3','L3'], xe[element_id] + 1)

        # Generate the mesh.
        mesh.generate(True) 

        # Refine the mesh along the xi1 direction
        refined_mesh = morphic.refinement.full_refinement(mesh, basis=["L3", "L3", "L3"], fac=2)

    """
    
    num_elem_nodes = get_num_elem_nodes(basis)
    xi_grid = generate_xi_grid(num_elem_nodes, dim=3)

    # Generate element partitions for each xi direction
    elem_partitions_xi1 = [(n / facs[0], (n + 1) / facs[0]) for n in range(facs[0])]
    elem_partitions_xi2 = [(n / facs[1], (n + 1) / facs[1]) for n in range(facs[1])]
    elem_partitions_xi3 = [(n / facs[2], (n + 1) / facs[2]) for n in range(facs[2])]

    # Combine the partitions for all xi directions
    xi_partitions = []
    for xi1_part in elem_partitions_xi1:
        for xi2_part in elem_partitions_xi2:
            for xi3_part in elem_partitions_xi3:
                xi_partitions.append(np.array([xi_grid[:, 0] * (xi1_part[1] - xi1_part[0]) + xi1_part[0],
                                               xi_grid[:, 1] * (xi2_part[1] - xi2_part[0]) + xi2_part[0],
                                               xi_grid[:, 2] * (xi3_part[1] - xi3_part[0]) + xi3_part[0]]).T)

    # Initialize mapping and refined element list
    mapping = {}
    refined_xe = []

    # Iterate over elements and xi partitions to refine the mesh
    for elem_idx, elem in enumerate(mesh.elements):
        for xi_part in xi_partitions:
            new_elem_nodes = elem.evaluate(xi_part)
            elem_node_ids = []
            for node in new_elem_nodes:
                # Check if point is already in the mapping, if not add it
                if tuple(node) not in mapping:
                    mapping[tuple(node)] = len(mapping)
                elem_node_ids.append(mapping[tuple(node)])
            refined_xe.append(elem_node_ids)

    # Generate the list of refined mesh nodes based on the mapping
    refined_xn_list = [list(point) for point in mapping.keys()]
    refined_xn = np.array(refined_xn_list) 
    refined_xe = np.array(refined_xe)

    # Update the refined mesh with the new nodes
    refined_mesh = create_morphic_mesh(basis, refined_xn, refined_xe)

    return refined_mesh, refined_xn, refined_xe