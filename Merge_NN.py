#!/usr/bin/env python
# coding: utf-8

import cdd
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import itertools
from dataclasses import dataclass, field
from typing import *
from abc import ABC, abstractmethod
import scipy
from scipy.spatial import ConvexHull
import matplotlib.cm as cm
import matplotlib.tri as tri
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.metrics import accuracy_score, log_loss
from collections import defaultdict

def save_weights(weights_list, biases_list, fname='weights_and_biases'):
    weights_and_biases_dict = {f'weights_{i}': weights for i, weights in enumerate(weights_list)}
    weights_and_biases_dict.update({f'biases_{i}': biases for i, biases in enumerate(biases_list)})
    weights_and_biases_dict.update({'number_layers': len(weights_list)})

    np.savez(f'{fname}.npz', **weights_and_biases_dict)

def load_weights(fname='weights_and_biases'):
    loaded = np.load(f'{fname}.npz')
    number_layers = loaded['number_layers']
    weights_list = [loaded[f'weights_{i}'] for i in range(0, number_layers)]
    biases_list = [loaded[f'biases_{i}'] for i in range(0, number_layers)]
    return weights_list, biases_list

def save_data(test_x, test_y, fname='data'):
    data_dict = {'test_x': test_x, 'test_y': test_y}
    np.savez(f'{fname}.npz', **data_dict)

def load_data(fname='data'):
    loaded = np.load(f'{fname}.npz')
    return loaded['test_x'], loaded['test_y']


# In[4]:


def compute_dimension(vertices):
    # Convert the vertices to a numpy array
    vertices_array = np.array(vertices)
    
    # Compute the rank of the matrix of vertices
    rank = np.linalg.matrix_rank(vertices_array)

    return rank

def vertices_to_equations(vertices):
    # first column in v_matrix is 1 to inducate vertices rather than rays
    v_matrix = np.hstack((np.ones((vertices.shape[0],1)), vertices))
    cdd_v_matrix = cdd.Matrix(np.array(v_matrix))
    vpoly = cdd.Polyhedron(cdd_v_matrix)
    equations = vpoly.get_inequalities()
    return equations

def merge_equations(equations_x, equations_y):
    return np.vstack((equations_x, equations_y))

def equations_to_vertices(equations):
    mat1 = cdd.Matrix(equations, number_type = 'fraction')
    mat1.rep_type = cdd.RepType.INEQUALITY
    poly1 = cdd.Polyhedron(mat1)

    gen = poly1.get_generators()

    if gen.row_size == 0:
        # empty matrix, activation pattern does not exist
        return None
    
    float_type = cdd.NumberTypeable('float')
    matrix = np.matrix([[float_type.make_number(gen[i][j]) for j in range(0,gen.col_size)] for i in range(0,gen.row_size)])    

    # check that cdd lib returned vertices (not rays)
    if not np.all(matrix[:,0] == 1):
        #import pdb; pdb.set_trace()
        #assert False
        # for some reason there is a ray. Possibly a rounding error? Ignore (treat as no solution)
        print(f"Warning; ray found in matrix {matrix}")
        # keep region for now (TODO: figure out what to do)

    # return just the vertices (not whether they are rays or not)
    return matrix[:,1:]

def intersection(vertices_x, vertices_y):
    equations_x = vertices_to_equations(vertices_x)
    equations_y = vertices_to_equations(vertices_y)
    equations_z = merge_equations(equations_x, equations_y)
    vertices_z = equations_to_vertices(equations_z)
    
    if vertices_z is None:
        return None

    # if edges(vertices_z) are not linearly independent, then return None (n-dimensioinal volume = 0)
    input_dims = vertices_z.shape[1] # number of coordinates in vertices is number of input dimensions
    if compute_dimension(vertices_z) < input_dims:
        # discard degenerate regions (points, lines etc. in higher dimensional space)
        return None
    
    return vertices_z

def avg_output(A, B):
    # function f is output within decison boundary A
    A_f = A.matrix_A
    B_f = A.vector_B
    # function g is output within decision boundary B
    A_g = B.matrix_A
    B_g = B.vector_B
    # output for region f is A_f * x + B_f
    # output for region g is A_g * x + B_g
    # return equation for region h (A_h * x + B_h) which is the average of f and g
    
    A_h = (A_f +A_g)/2
    B_h = (B_f +B_g)/2

    return A_h, B_h

def diff_output(A, B):
    # function f is output within decison boundary A
    A_f = A.matrix_A
    B_f = A.vector_B
    # function g is output within decision boundary B
    A_g = B.matrix_A
    B_g = B.vector_B
    # output for region f is A_f * x + B_f
    # output for region g is A_g * x + B_g
    # return equation for region h (A_h * x + B_h) which is the average of f and g
    
    A_h = (A_f - A_g)
    B_h = (B_f - B_g)

    return A_h, B_h

def volume_weighted_output(A, B):
    # function f is output within decison boundary A
    A_f = A.matrix_A
    B_f = A.vector_B
    vertices_f = A.vertices
    # function g is output within decision boundary B
    A_g = B.matrix_A
    B_g = B.vector_B
    vertices_g = B.vertices

    # output for region f is A_f * x + B_f
    # output for region g is A_g * x + B_g
    # return equation for region h (A_h * x + B_h) which is the average of f and g


    # Compute the Convex Hull of the vertices
    
    # Set flag Q12 to prevent the following:
    #
    # QH6271 qhull topology error (qh_check_dupridge): wide merge (824686947162.7x wider) due to dupridge between f3466 and f3563 (vertex dist 0.00039), merge dist 0.041, while processing p75
    # - Allow error with option 'Q12'
    # ...
    # A wide merge error has occurred.  Qhull has produced a wide facet due to facet merges and vertex merges.
    # This usually occurs when the input is nearly degenerate and substantial merging has occurred.
    # See http://www.qhull.org/html/qh-impre.htm#limit
    try:
        hull_f = ConvexHull(vertices_f, qhull_options = 'Q12')
        # The volume of the convex polytope is stored in the 'volume' attribute of the ConvexHull object
        volume_f = hull_f.volume
        hull_g = ConvexHull(vertices_g, qhull_options = 'Q12')
        volume_g = hull_g.volume

        weight_f = volume_f/(volume_f + volume_g) # depends on volume (scalar)
        weight_g = volume_g/(volume_f + volume_g)

        A_h = A_f * weight_f + A_g * weight_g
        B_h = B_f * weight_f + B_g * weight_g
    except scipy.spatial._qhull.QhullError as e:
        print(f"Warning: Caught QHullError: {e}")
        # couldn't do a volume merge for this region, so do a global average merge instead
        A_h = (A_f +A_g)/2
        B_h = (B_f +B_g)/2

    return A_h, B_h



#Given a polytope to check if the point is inside the polytope or outside, 
#we do a small check  inspired for mathematical analysis
#We take a small neighbourhood given a point in the $n$ dimensional space. For 2 dimensional polytopes the nbd is a
#square the side length $10^{-6}$. For 3 dimensional polytopes it's a cube of side length $10^{-6}$ and so on. 

def small_neighbourhood(input_point):
    #radius = 10**-6 # small value
    radius = 10**-3 # small value
    vertices = []
    input_dimensions = input_point.shape[0]
    for dim in range(input_dimensions):
        vertex1 = np.copy(input_point)
        vertex1[dim] -= radius
        vertices.append(vertex1)
        vertex2 = np.copy(input_point)
        vertex2[dim] += radius
        vertices.append(vertex2)
    return np.matrix(vertices)

# Example usage:
# example_point = np.array([0.5 , 0.5])
# small_neighbourhood(np.array([0.5,0.5,0.5]))
# print(intersection(example_region, small_neighbourhood(example_point)))

@dataclass
class DecisionBoundary(ABC):
    label: str
    
    @abstractmethod
    def intersection(self, Y):
        raise NotImplementedEror()
        
@dataclass
class DecisionSpace:
    boundaries: List[DecisionBoundary]

@dataclass
class DecisionBoundaryNN(DecisionBoundary):
    # y = A * x + B
    matrix_A: np.ndarray # A
    vector_B: np.ndarray # B
    vertices: np.ndarray # a numpy matrix of vertices defining the region     
    # number of items of each class in the region
    # density: List[int]
    
    def intersection(self, Y, merge_fn=avg_output):
        X = self
        vertices_x = X.vertices
        vertices_y = Y.vertices
        vertices_z = intersection(vertices_x, vertices_y)
        
        if vertices_z is None:
            return None
        
        label_z = X.label + "," + Y.label
        A_z, B_z = merge_fn(X, Y)
        return DecisionBoundaryNN(label_z, A_z, B_z, vertices_z)

@dataclass
class DecisionSpaceNN(DecisionSpace):
    pass


def in_boundary(boundary, input_vector):
    # return true if input_vector falls in boundary
    return intersection(boundary.vertices, small_neighbourhood(input_vector)) is not None
    
def find_boundary(boundaries, input_vector):
    for boundary in boundaries:
        if in_boundary(boundary, input_vector):
            return boundary
    # no boundary found (TODO figure out what to do!)

def get_output(decision_space, input_vector):
    # find decision boundary corresponding to input
    boundaries = decision_space.boundaries
    boundary = find_boundary(boundaries, input_vector)
    return get_boundary_output(boundary, input_vector)

def get_boundary_output(boundary, input_vector):
    # compute output for selected decision boundary
    A = boundary.matrix_A
    B = boundary.vector_B
    output_vector = A @ input_vector.T + B
    
    return output_vector[:,0].T

def vector_mask_to_matrix(v):
    return np.eye(v.shape[0]) * v


def find_activation_matrix_layer_l(coefs, intercepts, pattern_l, pattern_previous,
                                   A_previous, B_previous, layer,
                                   constraints_A, constraints_B):
    # pattern is a column vector, 1 => active, 0 => inactive
    # convert to                  1 => active, -1 => inactive (to make it easier to flip equations)
    inequality_dir = pattern_l * 2 - 1 # 1 -> 1, 0 -> -1
    #print("inequality_dir", inequality_dir)
    
    #constraints_A, constraints_b = get_constraints_layer1(pattern_layer1, coefs, intercepts)
    
    W_l = coefs[layer].transpose()
    
    # Convert pattern previous to matrix
    pattern_previous_matrix = vector_mask_to_matrix(pattern_previous)
    
    A_l = (W_l @ pattern_previous_matrix @ A_previous)
    
    # need to negate A_l as solves -Ax + b > 0
    A = np.vstack((-A_l * inequality_dir, constraints_A))
    
    b_l = intercepts[layer][:,np.newaxis]
    B_l = (W_l @ pattern_previous_matrix @ B_previous + b_l)

    b = np.vstack((B_l * inequality_dir, constraints_B))
    
    #import pdb; pdb.set_trace()
    
    # compute H = [b - A] to represent system of inequalities,
    # and use pycddlib to solve and find vertices of resultant polytope that satisfies inequalities
        
    H = np.hstack((b,-A))
    return H, A_l, B_l, A, b
    

def find_activation_region_layer_l(coefs, intercepts, pattern_l, pattern_previous,
                                   A_previous, B_previous, layer,
                                   constraints_A, constraints_B):
    
    H, A_l, B_l, A, b = find_activation_matrix_layer_l(coefs, intercepts, pattern_l, pattern_previous,
                                   A_previous, B_previous, layer,
                                   constraints_A, constraints_B)
    

    mat1 = cdd.Matrix(H, number_type = 'fraction')
    mat1.rep_type = cdd.RepType.INEQUALITY
    poly1 = cdd.Polyhedron(mat1)
    #print("H matrix", poly1) # debug
    
    gen = poly1.get_generators()
    #print(gen)
    #print("V matrix", gen) # debug

    
    if gen.row_size == 0:
        # empty matrix, activation pattern does not exist
        return None
    
    float_type = cdd.NumberTypeable('float')
    matrix = np.matrix([[float_type.make_number(gen[i][j]) for j in range(0,gen.col_size)] for i in range(0,gen.row_size)])    

    # check that cdd lib returned vertices (not rays)
    assert np.all(matrix[:,0] == 1)

    # return just the vertices (not whether they are rays or not)
    # also returns linear equation for region (A_l * x + B_l) and full set of constraints (A, b)
    return matrix[:,1:], A_l, B_l, A, b, pattern_l

def generate_regions(coefs, intercepts, layer):
    """
    Generates regions for the given layer
    coefs - weight coefs for NN generated by scikit-learn
    intercepts - biases for NN generated by scikit-learn
    layer - number of layers
    """

    def get_num_inputs():
        return coefs[0].shape[0]
        
    def get_neurons(l):
        return intercepts[l].shape[0]
    
    matricies = []

    # todo filter to just the activation patterns that have non-zero area
    perms_layers = []
    for l in range(layer):
        neurons_l = get_neurons(l)
        # todo: add one neuron at at time to avoid considering exponential number of permutations
        perms_l = list(itertools.product([0,1], repeat=neurons_l))
        perms_layers.append(perms_l)

    
    prev_results = []
    
    # 1st layer is special
    # assuming n layers
    for perm1 in perms_layers[0]:
        pattern_layer1 = np.array(perm1)[:,np.newaxis]

        # additional terms in A for constraints on x1, x2, ...:
        #x1 + 1 > 0
        #x2 + 1 > 0
        #x3 + 1 > 0
        # and also:
        #x1 - 1 < 0
        #x2 - 1 < 0
        #x3 - 1 < 0
        #..
        num_inputs = get_num_inputs()
        constraints_A_layer1 = np.vstack(
            (np.eye(num_inputs),
             -np.eye(num_inputs))
        )
        
        # additional terms in b for constraints on x1 and x2
        #x1 + 1 > 0
        #x2 + 1 > 0
        #x3 + 1 > 0
        #x1 - 1 < 0
        #x2 - 1 < 0
        #x3 - 1 < 0
        constraints_b_layer1 = np.ones((2 * num_inputs,1))
        
        neurons_layer1 = get_num_inputs()

        # layer 1 is index 0 in coefs
        # treat (non-existant) previous layer as all active neurons with identity
        result_1 = find_activation_region_layer_l(
                                    coefs, intercepts, pattern_layer1, np.ones(num_inputs),
                                    np.eye(num_inputs), np.zeros(num_inputs)[:,np.newaxis], 0,
                                    constraints_A_layer1, constraints_b_layer1)

        if result_1 is None:
            #print(f"could not find region for {perm1}")
            continue

        prev_results.append((result_1, [perm1]))
    
    # breadth-first iteration over all pattern results in each layer
    for l in range(1, layer):
        results = []
        
        for prev_result, prev_perms in prev_results:
            matrix_prev, A_lprev, B_lprev, A_prev, b_prev, pattern_layer_prev = prev_result
            
            for perm in perms_layers[l]:
                pattern_layer = np.array(perm)[:,np.newaxis]

                # layer 2 is index 1 in coefs
                result_l = find_activation_region_layer_l(
                                            coefs, intercepts, pattern_layer, pattern_layer_prev,
                                            A_lprev, B_lprev, l,
                                            A_prev, b_prev)

                if result_l is None:
                    #print(f"could not find region for {perm}")
                    continue
                
                results.append((result_l, prev_perms + [perm]))
        
        prev_results = results
    
    #matricies = []
    decision_boundaries = []
    
    for prev_result, prev_perms in prev_results:
        matrix, A_l, B_l, A_2, b_2, pattern = prev_result
        #matricies.append((matrix, str(prev_perms)))
        boundary = DecisionBoundaryNN(str(prev_perms), A_l, B_l, matrix)
        decision_boundaries.append(boundary)
    
    #return matricies
    return DecisionSpaceNN(decision_boundaries)


def relu(x):
    return np.maximum(0, x)

def find_pattern(coefs, intercepts, layer, z_previous):
    """
    z_previous - output of the previous layer (or input vector if is the first layer)
    Finds the activation pattern for a particular point
    """
    W_l = coefs[layer].transpose()
    b_l = intercepts[layer][:,np.newaxis]
    x_l = W_l @ z_previous + b_l
    z_l = relu(x_l)
    print(x_l) # debug
    pattern = tuple(np.where(x_l[:,0] > 0, 1, 0))
    return pattern, z_l, x_l
    


def find_region(coefs, intercepts, layer, input_point):
    """
    Finds activation region for the given input point
    coefs - weight coefs for NN generated by scikit-learn
    intercepts - biases for NN generated by scikit-learn
    layer - number of layers
    input_point - input point to lookup activation region for
    """

    def get_num_inputs():
        return coefs[0].shape[0]
        
    def get_neurons(l):
        return intercepts[l].shape[0]
    
    matricies = []
    empty_boundaries = []
    
    perms_layers = []
    z_previous = input_point[:, np.newaxis] # transform to column vector
    for l in range(layer):
        neurons_l = get_neurons(l)
        # just generate the pattern in each layer for the input_point
        pattern, z_previous, _ = find_pattern(coefs, intercepts, l, z_previous)
        #perms_layers.append([pattern[:,0]])
        perms_layers.append([pattern])
        #import pdb; pdb.set_trace()
    
    print(perms_layers)
    
    # rest of this function is identical to generate_regions
    prev_results = []
    
    # 1st layer is special
    # assuming n layers
    for perm1 in perms_layers[0]:
        pattern_layer1 = np.array(perm1)[:,np.newaxis]

        # additional terms in A for constraints on x1, x2, ...:
        #x1 + 1 > 0
        #x2 + 1 > 0
        #x3 + 1 > 0
        # and also:
        #x1 - 1 < 0
        #x2 - 1 < 0
        #x3 - 1 < 0
        #..
        num_inputs = get_num_inputs()
        constraints_A_layer1 = np.vstack(
            (np.eye(num_inputs),
             -np.eye(num_inputs))
        )
        
        # additional terms in b for constraints on x1 and x2
        #x1 + 1 > 0
        #x2 + 1 > 0
        #x3 + 1 > 0
        #x1 - 1 < 0
        #x2 - 1 < 0
        #x3 - 1 < 0
        constraints_b_layer1 = np.ones((2 * num_inputs,1))
        
        neurons_layer1 = get_num_inputs()

        # layer 1 is index 0 in coefs
        # treat (non-existant) previous layer as all active neurons with identity
        result_1 = find_activation_region_layer_l(
                                    coefs, intercepts, pattern_layer1, np.ones(num_inputs),
                                    np.eye(num_inputs), np.zeros(num_inputs)[:,np.newaxis], 0,
                                    constraints_A_layer1, constraints_b_layer1)

        if result_1 is None:
            H, _, _, _, _ = find_activation_matrix_layer_l(
                                    coefs, intercepts, pattern_layer1, np.ones(num_inputs),
                                    np.eye(num_inputs), np.zeros(num_inputs)[:,np.newaxis], 0,
                                    constraints_A_layer1, constraints_b_layer1)
            empty_boundaries.append(H)
            print(f"Layer 1: could not find region for {perm1}")
            continue

        prev_results.append((result_1, [perm1]))
    
    # breadth-first iteration over all pattern results in each layer
    for l in range(1, layer):
        results = []
        
        for prev_result, prev_perms in prev_results:
            matrix_prev, A_lprev, B_lprev, A_prev, b_prev, pattern_layer_prev = prev_result
            
            for perm in perms_layers[l]:
                pattern_layer = np.array(perm)[:,np.newaxis]

                # layer 2 is index 1 in coefs
                result_l = find_activation_region_layer_l(
                                            coefs, intercepts, pattern_layer, pattern_layer_prev,
                                            A_lprev, B_lprev, l,
                                            A_prev, b_prev)

                if result_l is None:
                    H, _, _, _, _ = find_activation_matrix_layer_l(coefs, intercepts, pattern_layer, pattern_layer_prev,
                                            A_lprev, B_lprev, l,
                                            A_prev, b_prev)
                    empty_boundaries.append(H)
                    print(f"Layer {l+1} could not find region for {perm}")
                    continue
                
                results.append((result_l, prev_perms + [perm]))
        
        prev_results = results
    
    #matricies = []
    decision_boundaries = []
    
    for prev_result, prev_perms in prev_results:
        matrix, A_l, B_l, A_2, b_2, pattern = prev_result
        #matricies.append((matrix, str(prev_perms)))
        boundary = DecisionBoundaryNN(str(prev_perms), A_l, B_l, matrix)
        decision_boundaries.append(boundary)
    
    #return matricies
    return DecisionSpaceNN(decision_boundaries), empty_boundaries


def find_pattern_point(coefs, intercepts, layer, input_point):
    """
    Finds pattern for the given input point
    coefs - weight coefs for NN generated by scikit-learn
    intercepts - biases for NN generated by scikit-learn
    layer - number of layers
    input_point - input point to lookup activation region for
    """

    perms_layers = []
    z_previous = input_point[:, np.newaxis] # transform to column vector
    for l in range(layer):
        # just generate the pattern in each layer for the input_point
        pattern, z_previous, _ = find_pattern(coefs, intercepts, l, z_previous)
        #perms_layers.append([pattern])
        perms_layers.append(pattern)        

    # pattern for all layers is a string of activation patterns of each layer
    return str(perms_layers)


def find_output_point(coefs, intercepts, layer, input_point):
    """
    Forward propogation to find the final output for a point
    coefs - weight coefs for NN generated by scikit-learn
    intercepts - biases for NN generated by scikit-learn
    layer - number of layers
    input_point - input point to lookup output for
    """

    z_previous = input_point[:, np.newaxis] # transform to column vector
    x_previous = z_previous
    for l in range(layer):
        # just generate the pattern in each layer for the input_point
        _, z_previous, x_previous = find_pattern(coefs, intercepts, l, z_previous)
        #perms_layers.append([pattern])

    # output of last layer (convert back to vector as only one input point)
    return x_previous[:,0]


def plot_decision_space_nn(decision_space : DecisionSpaceNN, output_dim=0, vis_out=True, shading='gouraud',
                          vmin=-7, vmax=7):
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Define a color map that linearly transitions from black to white
    cmap = LinearSegmentedColormap.from_list('black_to_white', ['black', 'white'], N=256)


    for boundary in decision_space.boundaries: # plot just first boundary
        vertices = np.array(boundary.vertices) # use np.array not np.matrix
        label =  boundary.label
        
        # compute the convex hull of the points
        hull = ConvexHull(vertices)

        # extract the x and y coordinates of the convex hull
        x = vertices[hull.vertices, 0]
        y = vertices[hull.vertices, 1]
        
        # output = A * input + B
        A = boundary.matrix_A
        B = boundary.vector_B
        
        if vis_out:
            input_vect = np.vstack((x.T, y.T))
            #import pdb; pdb.set_trace()
            output = A @ input_vect + B
        
            # Allow to select dimension (for now plot output dimension 0)
            output = output[output_dim]
            #print(max(output), min(output))

        # create a triangulation
        #import pdb; pdb.set_trace()
        #triang = tri.Triangulation(x.flatten(), y.flatten())

        # plot the triangles with colors
        #plt.tripcolor(triang, output, cmap=cm.viridis, alpha=0.3, edgecolors='k')
        #import pdb; pdb.set_trace()
        if vis_out:
            # shade triangle proportional to output
            plt.tripcolor(x.flatten(), y.flatten(), output, norm=norm, cmap=cmap, shading=shading)
        else:
            plt.fill(x, y, alpha=0.3, label=label, edgecolor='k')
            #plt.plot(x, y, 'ko')
            #plt.scatter(x, y, c=output, norm=norm, cmap=cmap)
                
        # If working, should always be 2 (same as input)
        # print(compute_dimension(vertices))

        #plt.plot(x, y, 'ko')

        # plot and fill the polygon
        #plt.fill(x, y, color=cm.viridis(output), alpha=0.3, label=label)

    # set the axis labels and title
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Polygon Plot')

    # show the plot with a colorbar
    if vis_out:
        plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()



def merge_decision_space(X, Y, merge_fn=avg_output):
    Z = []
    for boundary_x in X.boundaries:
        for boundary_y in Y.boundaries:
            intersection = boundary_x.intersection(boundary_y, merge_fn)
            if intersection is not None:
                Z.append(intersection)
    return DecisionSpace(Z)




def classify(decision_space, points):
    results = []
    for point in points:
        #print(f"Debug: classifying point {point}")
        y = get_output(decision_space, point)
        results.extend(y)
    return np.array(results)


def classify_on_fly(coefs, intercepts, layers, points):
    results = []
    for point in points:
        print(point)
        decision_space, _ = find_region(coefs, intercepts, layers, point)
        #print(f"Debug: classifying point {point}")
        y = get_boundary_output(decision_space.boundaries[0], point)
        results.append(y) # y can be multi-class
    return np.array(results)


def classify_on_fly_merged(coefs1, intercepts1, layers1, coefs2, intercepts2, layers2, merge_fn, points):
    results = []
    for point in points:
        print(point)
        decision_space1, _ = find_region(coefs1, intercepts1, layers1, point)
        decision_space2, _ = find_region(coefs2, intercepts2, layers2, point)
        # this sometimes results in rays / no region (todo: figure out why)
        # decision_space3 = merge_decision_space(decision_space1, decision_space2, merge_fn)
        
        X = decision_space1.boundaries[0] # only one boundary for find_region
        Y = decision_space2.boundaries[0]
        vertices_z = small_neighbourhood(point) # could be anything that contains point
        label_z = X.label + "," + Y.label
        A_z, B_z = merge_fn(X, Y)
        #decision_space3 = DecisionSpaceNN([DecisionBoundaryNN(label_z, A_z, B_z, vertices_z)])
        decision_boundary3 = DecisionBoundaryNN(label_z, A_z, B_z, vertices_z)

        #print(f"Debug: classifying point {point}")
        #y = get_output(decision_space3, point)
        y = get_boundary_output(decision_boundary3, point)
        #results.extend(y)
        results.append(y) # y can be multi-class
    return np.array(results)

def compute_volume(decision_boundary):
    vertices_f = decision_boundary.vertices
    try:
        hull_f = ConvexHull(vertices_f, qhull_options = 'Q12')
        # The volume of the convex polytope is stored in the 'volume' attribute of the ConvexHull object
        volume = hull_f.volume
    except scipy.spatial._qhull.QhullError as e:
        print(f"Warning: Caught QHullError: {e}")
        # couldn't do a volume merge for this region, so do a global average merge instead
        volume = float('nan')
    return volume

def tabulate_points_in_regions(coefs, intercepts, layers, points, compute_volume = True):
    boundariesX_to_points = defaultdict(list)
    boundariesX_to_volume = {}
    
    for point in points:
        print("processing training point")
        print(point)
        X_label = find_pattern_point(coefs, intercepts, layers, point)
        print(f"training point activation region is {X_label}")
        boundariesX_to_points[X_label].append(point)
        
        # add to volumes table:
        if X_label not in boundariesX_to_volume:
            if not compute_volume:
                # skip volume calculation
                boundariesX_to_volume[X_label] = float('nan')
                continue

            decision_space1, _ = find_region(coefs, intercepts, layers, point)
            X = decision_space1.boundaries[0] # only one boundary for find_region
            boundariesX_to_volume[X_label] = compute_volume(X)

    return boundariesX_to_points, boundariesX_to_volume

def extract_merge_attributes(coefs, intercepts, layers, training_points, points, do_volume_computation=True):
    # precompute number of training points in each region
    boundariesX_to_training_points, _ = tabulate_points_in_regions(coefs, intercepts, layers, training_points, compute_volume = False)

    # used to cache boundaries if examining the same region twice
    label_to_boundary = {}
    # used to cache volume if examining the same region twice
    label_to_volume = {}
    results_output = []
    results_volume = []
    results_num_training_points = []
    
    print("calling extract merge attributes")
    print("points:", points)

    for point in points:
        print("running loop")
        print(point)
        X_label = find_pattern_point(coefs, intercepts, layers, point)
        if do_volume_computation == True:
            # add to volumes table:
            if X_label not in label_to_boundary:
                decision_space1, _ = find_region(coefs, intercepts, layers, point)
                X = decision_space1.boundaries[0] # only one boundary for find_region
                label_to_boundary[X_label] = X
            
            if X_label not in label_to_volume:
                label_to_volume[X_label] = compute_volume(label_to_boundary[X_label])
            
            # output volume attribute for region point is in
            volume = label_to_volume[X_label]
            results_volume.append(volume)
        else:
            volume = float('nan')
            results_volume.append(volume)
            
        if X_label in boundariesX_to_training_points:
            num_training_points = len(boundariesX_to_training_points[X_label])
        else:
            # no training points in this region
            num_training_points = 0

        # output no training points for region point is in
        results_num_training_points.append(num_training_points)

        # output prediction for point
        # (when we don't need volume, we don't have boundaries, so run traditional forward propogation instead)
        #pred_y = get_boundary_output(label_to_boundary[X_label], point)
        pred_y = find_output_point(coefs, intercepts, layers, point)
        print("pred_y", pred_y)

        results_output.append(pred_y)

    return results_output, results_volume, results_num_training_points

def nn_accuracy(pred_y, test_y):
    # pred_y - output (prior to softmax) of neural network
    # test_y - the true classes
    #
    # convert to 0 or 1 prediction (assuming a single class)
    pred_y = np.where(pred_y[:,0] > 0, 1, 0)
    acc = accuracy_score(test_y, pred_y)
    return acc

def nn_log_loss(pred_y, test_y):
    # pred_y - output (prior to softmax) of neural network
    # test_y - the true classes
    #
    # Calculate output after softmax (between 0 and 1) (assuming a single class)
    # TODO: Is this a bug? Should use sigmoid instead?
    pred_y = scipy.special.softmax(pred_y)
    loss = log_loss(test_y, pred_y)
    return loss
