.. _install-developers:
.. _building-cmake:
.. _building-cmake-intro:

Developers
==========

`CMake <https://cmake.org>`_ is our primary build system.
If you are new to CMake, `this short tutorial <https://hsf-training.github.io/hsf-training-cmake-webpage/>`_ from the HEP Software foundation is the perfect place to get started.
If you just want to use CMake to build the project, jump into sections `1. Introduction <https://hsf-training.github.io/hsf-training-cmake-webpage/01-intro/index.html>`__, `2. Building with CMake <https://hsf-training.github.io/hsf-training-cmake-webpage/02-building/index.html>`__ and `9. Finding Packages <https://hsf-training.github.io/hsf-training-cmake-webpage/09-findingpackages/index.html>`__.

Dependencies
------------

Before you start, you will need a copy of the ExaEpi source code:

.. code-block:: bash

   git clone https://github.com/AMReX-Codes/exaepi.git $HOME/src/exaepi
   cd $HOME/src/exaepi

ExaEpi depends on popular third party software.

* On your development machine, :ref:`follow the instructions here <install-dependencies>`.
* If you are on an HPC machine, :ref:`follow the instructions here <install-hpc>`.

.. toctree::
   :hidden:

   dependencies

Compile
-------

From the base of the ExaEpi source directory, execute:

.. code-block:: bash

   # find dependencies & configure
   #   see additional options below, e.g.
   #                   -DCMAKE_INSTALL_PREFIX=$HOME/sw/exaepi
   cmake -S . -B build

   # compile, here we use four threads
   cmake --build build -j 4

That's all! ExaEpi binaries are now in ``build/bin/``.
You can execute these binaries directly or copy them out.

You can inspect and modify build options after running ``cmake -S . -B build`` with either

.. code-block:: bash

   ccmake build

or by adding arguments with ``-D<OPTION>=<VALUE>`` to the first CMake call, e.g.:

.. code-block:: bash

   cmake -S . -B build -DAMReX_COMPUTE=CUDA
