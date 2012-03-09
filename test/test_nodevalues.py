import sys
import unittest
import doctest

import numpy
from numpy import array
import numpy.testing as npt

sys.path.append('..')
import core
import mesher

class TestNodeValues(unittest.TestCase):
    """Unit tests for morphic NodeValues class."""
    
    def test_nodevalues_init(self):
        mesh = mesher.Mesh()
        Xn = numpy.array([[0, 0.1, 0.2], [0.3, 0.4, 0.5]])
        node = mesh.add_stdnode(1, Xn)
        npt.assert_equal(node.values, Xn)
        npt.assert_equal(node.values[:, 0], Xn[:, 0])
        npt.assert_equal(node.values[1, :], Xn[1, :])
        npt.assert_equal(node.values[1, 1], Xn[1, 1])
        npt.assert_equal(node.values[:, :2], Xn[:, :2])
        npt.assert_equal(mesh._core.P, Xn.flatten())
    
    def test_set_all(self):
        mesh = mesher.Mesh()
        Xn = numpy.array([[0, 0.1, 0.2], [0.3, 0.4, 0.5]])
        node = mesh.add_stdnode(1, Xn)
        npt.assert_equal(mesh._core.P, Xn.flatten())
        npt.assert_equal(node.values, Xn)
        
        Xn2 = numpy.array([[11, 12, 13], [15, 16, 66]])
        node.values = numpy.array([[11., 12, 13], [15, 16, 66]])
        npt.assert_equal(mesh._core.P, Xn2.flatten())
        npt.assert_equal(node.values, Xn2)
        
    def test_set_all_different_shape(self):
        mesh = mesher.Mesh()
        Xn = numpy.array([[0, 0.1, 0.2], [0.3, 0.4, 0.5]])
        node = mesh.add_stdnode(1, Xn)
        
        Xn2 = numpy.array([[11, 12], [15, 16], [22, 33]])
        self.assertRaises(IndexError, self.assign_function, node, Xn2)
    
    def assign_function(self, node, Xn2):
        node.values = Xn2
    
    def test_set_slices(self):
        mesh = mesher.Mesh()
        Xn1 = numpy.array([[0, 0.1, 0.2], [0.3, 0.4, 0.5]])
        Xn2 = numpy.array([[11, 12, 13], [15, 16, 66]])
        P = numpy.append(Xn1.flatten(), Xn2.flatten())
        node1 = mesh.add_stdnode(1, Xn1)
        node2 = mesh.add_stdnode(2, Xn2)
        
        npt.assert_equal(mesh._core.P, P)
        npt.assert_equal(node1.values, Xn1)
        npt.assert_equal(node2.values, Xn2)
        
        x = [99, 33]
        Xn1[:, 0], P[[0, 3]], node1.values[:, 0] = x, x, x
        npt.assert_equal(mesh._core.P, P)
        npt.assert_equal(node1.values, Xn1)
        
        x = [11, 22]
        Xn2[1, 1:], P[[10, 11]], node2.values[1, 1:] = x, x, x
        npt.assert_equal(mesh._core.P, P)
        npt.assert_equal(node2.values, Xn2)
        
        x = 11
        Xn2[1, 0], P[9], node2.values[1, 0] = x, x, x
        npt.assert_equal(mesh._core.P, P)
        npt.assert_equal(node2.values, Xn2)
        
        x = [23, 45]
        Xn1[0, -2:], P[[1, 2]], node1.values[0, -2:] = x, x, x
        npt.assert_equal(mesh._core.P, P)
        npt.assert_equal(node1.values, Xn1)
        
        
        
if __name__ == "__main__":
    unittest.main()