Complete documentation
#######################################

.. _class_d_vector:

d_vector
==============

This class creates and manages an array on the device (`aka.` on the GPU). 

Properties
***********

+---------+------------------+-----------------------------------------------------+
| int     | length           | The length of the array. It cannot be resized.      |
+---------+------------------+-----------------------------------------------------+

Methods
*********

d_vector(int n)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor creates an array of size `n` on the device.

float at(int i)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the i'th element of the array.

float dot(d_vector other)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the dot product of this d_vector and the `other` d_vector.

________________________________________________________

.. _class_d_spmatrix:

d_spmatrix
==============

This class creates and manages a sparse matrix the device (`aka.` on the GPU). 

Properties
***********

+-------------------------------------+---------------+--------------------------------------------------------------+
| (int, int)                          | shape         | A (rows, columns) tuple representing the shape of the matrix.|
+-------------------------------------+---------------+--------------------------------------------------------------+
|        int                          | nnz           | Number of non-zero elements in the matrix.                   |
+-------------------------------------+---------------+--------------------------------------------------------------+
|:ref:`matrix_type<class_matrix_type>`| dtype         | Number of non-zero elements in the matrix.                   |
+-------------------------------------+---------------+--------------------------------------------------------------+


Methods
*********

d_spmatrix(int rows, int columns, int nnz = 0, :ref:`matrix_type<class_matrix_type>` type = COO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor creates a sparse matrix of shape (`rows`, `columns`)
with `nnz` non-zero elements on the device.

void to_csr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts a COO matrix into a CSR matrix.

________________________________________________________

.. _class_matrix_type:

matrix_type
==============

An enum that represents the data storage type of a sparse matrix.


Values
***********

+---+--------+
| 0 | COO    | 
+---+--------+
| 1 | CSR    |
+---+--------+
| 2 | CSC    |
+---+--------+

________________________________________________________

.. _class_state:

state
===============

A class for managing the concentrations of species. 
A `state` will create and hold a :ref:`d_vector<class_d_vector>` for each species you add.

Properties
***********

+---------+------------------+-----------------------------------------------------+
| int     | n_species        | Number of species                                   |
+---------+------------------+-----------------------------------------------------+
| int     | vector_size      | Number of spatial nodes for each species            |
+---------+------------------+-----------------------------------------------------+

Methods
*********

state (int vector_size)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor creates a `state` that will hold `vector_size` sized species.

add_species (string name, bool diffusion = true)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a species called `name` to the state. Set `diffusion` to `False` 
if you don't want the species to be affected by diffusion.

:ref:`d_vector<class_d_vector>` get_species (string name)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a reference to the current vector of the given species.

Returns NULL if the species is not found.
 
.. _method_set_species:

void set_species (string name, :ref:`d_vector<class_d_vector>` species_vector)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copies the given vector as the new state-vector of the given species.

If the species is not found, it does nothing.

void set_species (string name, numpy.array species_vector)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overloads the :ref:`above method<method_set_species>`.

You can specify the vector as numpy array.


void print (int printCount = 5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prints each species and its vector. 

For each vector, only `printCount` elements will be printed.


void list_species ()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the species of the `state` as a list of strings.

________________________________________________________

.. _class_simulation:


simulation
===============

A class for managing simulations. 
A `simulation` will create and hold a :ref:`state<class_state>`.

Properties
***********
+--------------------------+------------------+---------------------------------------------------------------------------------------------------------------+
| :ref:`state<class_state>`| current_state    | The :ref:`state<class_state>` object held by the simulation                                                   |
+--------------------------+------------------+---------------------------------------------------------------------------------------------------------------+
| float                    | epsilon          | The error of the conjugate gradient method. Defaults at 10e-3                                                 |
+--------------------------+------------------+---------------------------------------------------------------------------------------------------------------+
| float                    | drain            | The drain is a constant value that is deducted from the concentration of each species after each reaction-step|
+--------------------------+------------------+---------------------------------------------------------------------------------------------------------------+


Methods
*********

simulation (int vector_size)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor creates a :ref:`simulation<class_simulation>` and a :ref:`state<class_state>`
that holds `vector_size` sized species.

add_species (string name, bool diffusion = true)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a species called `name` to the current_state. Set `diffusion` to `False` 
if you don't want the species to be affected by diffusion.

:ref:`d_vector<class_d_vector>` get_species (string name)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a reference to the current vector of the given species.

Returns NULL if the species is not found.
 
.. _method_set_species_2:

void set_species (string name, :ref:`d_vector<class_d_vector>` species_vector)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copies the given vector as the new state-vector of the given species.

If the species is not found, it does nothing.

void set_species (string name, numpy.array species_vector)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overloads the :ref:`above method<method_set_species_2>`.

You can specify the vector as numpy array.

.. _method_add_reaction:

void add_reaction (string reaction, float rate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds the given reaction to the simulation. It will be taken into account in all future reaction-steps.

Some examples of reactions that can be written:

``"A + B -> C"``

``" B+C-> "``

Note that this method will raise an error if any of the mentioned species has not be added beforehand. 


void add_reversible_reaction (string reaction, float rate_forward, float rate_back)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the :ref:`add_reaction<method_add_reaction>` method. 
Adds both the given reaction and its reverse.
The rates for each of the two reactions has to be specified. 

void load_dampness_matrix (:ref:`d_spmatrix<class_d_spmatrix>` dampness_matrix)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sets the given matrix as the reactor's dampness matrix. Mandatory for performing diffusion.

void load_stiffness_matrix (:ref:`d_spmatrix<class_d_spmatrix>` stiffness_matrix)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sets the given matrix as the reactor's stiffness matrix. Mandatory for performing diffusion.

___________________________________________________________________________________________________________

