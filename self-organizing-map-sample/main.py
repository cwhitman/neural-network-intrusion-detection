import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

"""
A simple test of a self organizing map neural network.
This article: https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
was used as a guide. It was updated to match the current version of tensorflow.
"""

class SOM(object):

    # To check if the SOM has been trained
    _trained = False

    """
        Args:
            m: length of map
            n: width of map
            dim: dimension of training inputs
            n_iterations: total number of iterations
            alpha: Learning rate
            sigma: Initial neighborhood value. Defined to be max(n,m)/2 if not set.
    """
    def __init__(self, m, n, dim, n_iterations=100, alpha=0.3, sigma=None):

        #initializing values
        self._m = m
        self._n = n
        if sigma is None:
            sigma = max(m, n) / 2.0
        sigma = float(sigma)
        alpha = float(alpha)
        self._n_iterations = abs(int(n_iterations))

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m * n, dim],dtype=tf.float32),dtype=tf.float32)

            # Matrix of size [m*n, 2] for SOM grid locations
            # of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            # We need to assign them as attributes to self, since they
            # will be fed in during training

            # The training vector
            self._vect_input = tf.placeholder(tf.float32, [dim])
            # Iteration number
            self._iter_input = tf.placeholder(tf.float32)

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            #Will need to change if we ever use something besides Euclidean distance
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m * n)])), 2), 1)),0)

            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]),dtype=tf.int64)),[2])

            # To compute the alpha and sigma values based on iteration
            # number
            #Both alpha and sigma are defined by
            # x= x_0 * (1-i/n) where x is alpha/sigma, x_0 is the initial value, i is the current iteration, and n is
            #the total number of iterations.
            #Will need to mess with this rate when we implement this.
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)


            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            #learning rates decrease with time
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m * n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, tf.float32), tf.cast(tf.pow(_sigma_op, 2),tf.float32))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input

            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m * n)]),
                       self._weightage_vects))

            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)

            self._training_op = tf.assign(self._weightage_vects,new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    """
    Trains the SOM.
    'input_vects' should be an iterable of 1-D NumPy arrays with
    dimensionality as provided during initialization of this SOM.
    Current weightage vectors for all neurons(initially random) are
    taken as starting conditions for training.
    """
    def train(self, input_vects):
        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    """
    Maps each input vector to the relevant neuron in the SOM
    grid.
    'input_vects' should be an iterable of 1-D NumPy arrays with
    dimensionality as provided during initialization of this SOM.
    Returns a list of 1-D NumPy arrays containing (row, column)
    info for each input vector(in the same order), corresponding
    to mapped neuron.
    """
    def map_vects(self, input_vects):

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return

    """
    Yields one by one the 2-D locations of the individual neurons
    in the SOM.
    """
    def _neuron_locations(self, m, n):

        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])


def main():
    # Training inputs for RGBcolors
    colors = np.array(
        [[0., 0., 0.],
         [0., 0., 1.],
         [0., 0., 0.5],
         [0.125, 0.529, 1.0],
         [0.33, 0.4, 0.67],
         [0.6, 0.5, 1.0],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 1., 1.],
         [1., 0., 1.],
         [1., 1., 0.],
         [1., 1., 1.],
         [.33, .33, .33],
         [.5, .5, .5],
         [.66, .66, .66]])
    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']

    # Train a 20x30 SOM with 400 iterations
    som = SOM(20, 30, 3, 400)
    som.train(colors)

    # Get output grid
    image_grid = som.get_centroids()

    # Map colours to their closest neurons
    mapped = som.map_vects(colors)

    # Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()

if __name__=="__main__":
    main()
