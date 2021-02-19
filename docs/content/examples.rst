Examples
#######################################


One species in a square
=========================

In this simulation, we use a single species (no reactions) placed in a square-shaped reactor.

We place an initial concentration of the species in a corner of the reactor.

.. image:: images/square_onespecies_t=0.png

Here is what the siulation should output after a few iterations of diffusion.

.. image:: images/square_onespecies_t=10.png



Use WolframScript to create new reactor shapes
================================================

WolframScript is a free tool for developpers. It allows you to use the Wolfram Engine via the command line.
Once installed, you should have access to the ``` wolframscript ``` command.

Two scripts are provided in the wolframScripts folder:

        ImageToData.wls
        MazeGenerator.wls

Image to data
**************

First, you need to have black and white image, where the white represents the area of your reactor, ad in the exemple below:

.. image:: images/maze.png

Run the ImageToData script using the wolframscript command::
    
    wolframscript /path/to/ImageToData.wls /path/to/image.png

For example, you can run this command from the ARDiS directory::

    wolframscript wolframScripts/ImageToData.wls data/maze.png


Maze generator
***************

This script generates a random maze at a given path::

    wolframscript /path/to/MazeGenerator.wls /given/path size

For example::

    wolframscript wolframScripts/MazeGenerator.wls data/myMaze 4
