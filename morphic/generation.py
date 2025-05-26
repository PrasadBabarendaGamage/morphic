import numpy as np  
import copy 

def generate_element_nodes(basis, num_elem_xi, extent=[]):
    A = 1
    if basis == ['H3', 'H3']:
        BASIS_NUMBER_OF_NODES = 4
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [2, 2]
        NUMBER_OF_NODES_XIC.insert(0, 0)
    elif basis == ['L2', 'L2']:
        BASIS_NUMBER_OF_NODES = 9
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [3, 3]
        NUMBER_OF_NODES_XIC.insert(0, 0)
    elif basis == ['L3', 'L3']:
        BASIS_NUMBER_OF_NODES = 16
        NUMBER_OF_XIC = 2
        NUMBER_OF_NODES_XIC = [4, 4]
        NUMBER_OF_NODES_XIC.insert(0, 0)
    elif basis == ['L3', 'L3', 'L3']:
        BASIS_NUMBER_OF_NODES = 64
        NUMBER_OF_XIC = 3
        NUMBER_OF_NODES_XIC = [4, 4, 4]
        NUMBER_OF_NODES_XIC.insert(0, 0)
    elif basis == ['L5', 'L5', 'L5']:
        BASIS_NUMBER_OF_NODES = 6 * 6 * 6
        NUMBER_OF_XIC = 3
        NUMBER_OF_NODES_XIC = [6, 6, 6]
        NUMBER_OF_NODES_XIC.insert(0, 0)

    NUMBER_OF_ELEMENTS_XIC = copy.deepcopy(num_elem_xi)
    NUMBER_OF_ELEMENTS_XIC.insert(0, 0)
    TOTAL_NUMBER_OF_NODES_XIC = [0] * (3)
    TOTAL_NUMBER_OF_NODES_XIC.insert(0, 0)
    TOTAL_NUMBER_OF_ELEMENTS_XIC = [0] * (3)
    TOTAL_NUMBER_OF_ELEMENTS_XIC.insert(0, 0)

    # Calculate sizes
    TOTAL_NUMBER_OF_NODES = 1
    TOTAL_NUMBER_OF_ELEMENTS = 1
    for xic_idx in range(1, NUMBER_OF_XIC + A):
        TOTAL_NUMBER_OF_NODES_XIC[xic_idx] = (NUMBER_OF_NODES_XIC[
                                                  xic_idx] - 2) * \
                                             NUMBER_OF_ELEMENTS_XIC[xic_idx] + \
                                             NUMBER_OF_ELEMENTS_XIC[
                                                 xic_idx] + 1
        TOTAL_NUMBER_OF_ELEMENTS_XIC[xic_idx] = NUMBER_OF_ELEMENTS_XIC[xic_idx]
        TOTAL_NUMBER_OF_NODES = TOTAL_NUMBER_OF_NODES * \
                                TOTAL_NUMBER_OF_NODES_XIC[xic_idx]
        TOTAL_NUMBER_OF_ELEMENTS = TOTAL_NUMBER_OF_ELEMENTS * \
                                   TOTAL_NUMBER_OF_ELEMENTS_XIC[xic_idx]

    Xe_nodes = np.zeros((TOTAL_NUMBER_OF_ELEMENTS, BASIS_NUMBER_OF_NODES),
                           dtype='uint32')
    Xeid = 0
    # !Set the elements for the regular mesh
    ELEMENT_NODES = [0] * (BASIS_NUMBER_OF_NODES + A)
    # Step in the xi[3)direction
    for ne3 in range(A, TOTAL_NUMBER_OF_ELEMENTS_XIC[3] + 1 + A):
        for ne2 in range(A, TOTAL_NUMBER_OF_ELEMENTS_XIC[2] + 1 + A):
            for ne1 in range(A, TOTAL_NUMBER_OF_ELEMENTS_XIC[1] + 1 + A):
                if ((NUMBER_OF_XIC < 3) or (
                        ne3 <= TOTAL_NUMBER_OF_ELEMENTS_XIC[3])):
                    if (NUMBER_OF_XIC < 2 or ne2 <=
                            TOTAL_NUMBER_OF_ELEMENTS_XIC[2]):
                        if (ne1 <= TOTAL_NUMBER_OF_ELEMENTS_XIC[1]):
                            ne = ne1
                            npp = 1 + (ne1 - 1) * (NUMBER_OF_NODES_XIC[1] - 1)
                            if (NUMBER_OF_XIC > 1):
                                ne = ne + (ne2 - 1) * \
                                     TOTAL_NUMBER_OF_ELEMENTS_XIC[1]
                                npp = npp + (ne2 - 1) * \
                                     TOTAL_NUMBER_OF_NODES_XIC[1] * (
                                                 NUMBER_OF_NODES_XIC[2] - 1)
                                if (NUMBER_OF_XIC > 2):
                                    ne = ne + (ne3 - 1) * \
                                         TOTAL_NUMBER_OF_ELEMENTS_XIC[1] * \
                                         TOTAL_NUMBER_OF_ELEMENTS_XIC[2]
                                    npp = npp + (ne3 - 1) * \
                                         TOTAL_NUMBER_OF_NODES_XIC[1] * \
                                         TOTAL_NUMBER_OF_NODES_XIC[2] * (
                                                     NUMBER_OF_NODES_XIC[
                                                         3] - 1)
                            nn = 0
                            for nn1 in range(A, NUMBER_OF_NODES_XIC[1] + A):
                                nn = nn + 1
                                ELEMENT_NODES[nn] = npp + (nn1 - 1)
                            if (NUMBER_OF_XIC > 1):
                                for nn2 in range(A + 1,
                                                 NUMBER_OF_NODES_XIC[2] + A):
                                    for nn1 in range(A, NUMBER_OF_NODES_XIC[
                                                            1] + A):
                                        nn = nn + 1
                                        ELEMENT_NODES[nn] = npp + (nn1 - 1) + (
                                                    nn2 - 1) * \
                                                            TOTAL_NUMBER_OF_NODES_XIC[
                                                                1]
                                if (NUMBER_OF_XIC > 2):
                                    for nn3 in range(A + 1,
                                                     NUMBER_OF_NODES_XIC[
                                                         3] + A):
                                        for nn2 in range(A,
                                                         NUMBER_OF_NODES_XIC[
                                                             2] + A):
                                            for nn1 in range(A,
                                                             NUMBER_OF_NODES_XIC[
                                                                 1] + A):
                                                nn = nn + 1
                                                ELEMENT_NODES[nn] = npp + (
                                                            nn1 - 1) + (
                                                                                nn2 - 1) * \
                                                                    TOTAL_NUMBER_OF_NODES_XIC[
                                                                        1] + (
                                                                                nn3 - 1) * \
                                                                    TOTAL_NUMBER_OF_NODES_XIC[
                                                                        1] * \
                                                                    TOTAL_NUMBER_OF_NODES_XIC[
                                                                        2]
                            # print(ELEMENT_NODES[1:len(ELEMENT_NODES)])
                            Xe_nodes[Xeid, :] = ELEMENT_NODES[
                                                1:len(ELEMENT_NODES)]
                            Xeid += 1

    if extent == []:
        return Xe_nodes - 1  # Generated node numbers begin from 0
    else:
        Xn = np.zeros((TOTAL_NUMBER_OF_NODES, NUMBER_OF_XIC))

        FIELD_NODE_USER_NUMBER = 0

        if NUMBER_OF_XIC == 3:
            VALUE = [0.0] * (3)
            for Z_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[3]):
                for Y_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[2]):
                    for X_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[1]):
                        for XIC_COORDINATE in range(NUMBER_OF_XIC):
                            Xn[FIELD_NODE_USER_NUMBER, XIC_COORDINATE] = float(
                                VALUE[XIC_COORDINATE])
                        FIELD_NODE_USER_NUMBER += 1
                        VALUE[0] = float(VALUE[0]) + float(extent[0]) / float(
                            (TOTAL_NUMBER_OF_NODES_XIC[1] - 1))
                    VALUE[1] = float(VALUE[1]) + float(extent[1]) / float(
                        (TOTAL_NUMBER_OF_NODES_XIC[2] - 1))
                    VALUE[0] = 0.0
                VALUE[2] = float(VALUE[2]) + float(extent[2]) / float(
                    (TOTAL_NUMBER_OF_NODES_XIC[3] - 1))
                VALUE[1] = 0.0
        elif NUMBER_OF_XIC == 2:
            VALUE = [0] * (2)
            for Y_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[2]):
                for X_COUNTER in range(TOTAL_NUMBER_OF_NODES_XIC[1]):
                    for XIC_COORDINATE in range(NUMBER_OF_XIC):
                        Xn[FIELD_NODE_USER_NUMBER, XIC_COORDINATE] = VALUE[
                            XIC_COORDINATE]
                    FIELD_NODE_USER_NUMBER += 1
                    VALUE[0] = VALUE[0] + (
                                extent[0] / (TOTAL_NUMBER_OF_NODES_XIC[1] - 1))
                VALUE[1] = extent[1] / (TOTAL_NUMBER_OF_NODES_XIC[2] - 1)
                VALUE[0] = 0

        return Xe_nodes - 1, Xn  # Generated node numbers begin from 0 