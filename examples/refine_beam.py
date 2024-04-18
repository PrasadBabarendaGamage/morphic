import morphic

xe, xn = morphic.generation.generate_element_nodes(['L3', 'L3', 'L3'], [2, 2, 2], extent=[1, 1, 1])

mesh = morphic.Mesh()

# Add nodes.
for node_id, value in enumerate(xn):
    mesh.add_stdnode(node_id + 1, value, group='_default')

# Add elements.
for element_id, element in enumerate(xe):
    mesh.add_element(element_id + 1, ['L3', 'L3','L3'], xe[element_id] + 1)

# Generate the mesh.
mesh.generate(True) 

# Refine the mesh along the xi1 direction. 
refined_mesh_x1 = morphic.refinement.mesh_refine_along_xi1(mesh, basis=['L3', 'L3', 'L3'], fac=2) 

# Refine the mesh along the xi2 direction. 
refined_mesh_x2 = morphic.refinement.mesh_refine_along_xi2(mesh, basis=['L3', 'L3', 'L3'], fac=2)

# Refine the mesh along the xi3 direction. 
refined_mesh_x3 = morphic.refinement.mesh_refine_along_xi3(mesh, basis=['L3', 'L3', 'L3'], fac=2) 

# Refine along all direction at once. 
refined_mesh = morphic.refinement.full_refinement(mesh, basis=['L3', 'L3', 'L3'], facs=[2, 2, 2])  