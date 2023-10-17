.. _usage_run:

Run ExaEpi
==========

In order to run a new simulation:

#. create a **new directory**, where the simulation will be run
#. make sure the ExaEpi **executable** is either copied into this directory or in your ``PATH`` `environment variable <https://en.wikipedia.org/wiki/PATH_(variable)>`__
#. add an **inputs file** and on :ref:`HPC systems <install-hpc>` a **submission script** to the directory
#. run

.. code-block:: bash

   cd <run_directory>

   # run with an inputs file:
   mpirun -np <n_ranks> ./agent <input_file>

On an :ref:`HPC system <install-hpc>`, you would instead submit the :ref:`job script <install-hpc>` at this point, e.g. ``sbatch <submission_script>`` (SLURM on Cori/NERSC) or ``bsub <submission_script>`` (LSF on Summit/OLCF).

Inputs Parameters
=================

Runtime parameters are specified in an `inputs` file, which is required to run ExaEpi.
An example `inputs` file can be bound at `ExaEpi/examples/inputs`. Below, we document
the runtime parameters than can be set in the inputs file.

In addition to the ExaEpi inputs, there are also a number of runtime options that can be configured for AMReX itself. Please see XXX for more information on these options.

* ``agents.size`` (`tuple of 2 integers`: e.g. ``(1, 1)``)
    This option is deprecated and will removed in a future version of ExaEpi. It controls
    the number of cells in the domain when running in `demo` mode. During actual usage,
    this number will be overridden and is irrelevant.
* ``agent.max_grid_size`` (`integer`)
    This option sets the maximum grid size used for MPI domain decomposition. If set to
    ``16``, for example, the domain will be broken up into grids of `16^2` communities, and
    these grids will be assigned to different MPI ranks / GPUs.
* ``agent.ic_type`` (`string`: either ``"census"`` or ``"demo"``)
    If ``"census"``, initial conditions will be read from the provided census data file.
    If ``"demo"``, agents will be initialized according to a power law distribution.
    Note that the ``"demo"`` `ic_type` is deprecated and will be removed in the future.
* ``agent.census_filename`` (`string`)
    The path to the ``*.dat`` file containing the census data used to set initial conditions.
    Must be provided if ``ic_type`` is ``"census"``. Examples of these data files are provided
    in ``ExaEpi/data/CensusData``.
* ``agent.worker_filename`` (`string`)
    The path to the ``*.bin`` file containing worker flow information.
    Must be provided if ``ic_type`` is ``"census"``. Examples of these data files are provided
    in ``ExaEpi/data/CensusData``.
* ``agent.case_filename`` (`string`)
    The path to the ``*.cases`` file containing the initial case data to use.
    Must be provided if ``ic_type`` is ``"census"``. Examples of these data files are provided
    in ``ExaEpi/data/CaseData``.
* ``agent.nsteps`` (`integer`)
    The number of time steps to simulate. Currently, time steps are fixed at 12 hours, so to
    run for 30 days, input `60`.
* ``agent.plot_int`` (`integer`)
    The number of time steps between successive plot file writes.
* ``agent.random_travel_int`` (`integer`)
    The number of time steps between long distance travel events - note that this is
    currently only meaningful if `ic_type` = ``"census"``.
* ``agent.aggregated_diag_int``
    The number of time steps between writing aggregated data, for example wastewater data.
* ``agent.aggregated_diag_prefix`` (`string`)
    Prefix to use when writing aggregated data. For example, if this is set to `cases`, the
    aggregated data files will be named `cases000010`, etc.
* ``contact.pSC`` (`float`, default: 0.2)
    This is contact matrix scaling factor for schools.
* ``contact.pCO`` (`float`, default: 1.45)
    This is contact matrix scaling factor for communities.
* ``contact.pNH`` (`float`, default: 1.45)
    This is contact matrix scaling factor for neighborhoods.
* ``contact.pWO`` (`float`, default: 0.5)
    This is contact matrix scaling factor for workplaces.
* ``contact.pFA`` (`float`, default: 1.0)
    This is contact matrix scaling factor for families.
* ``contact.pBAR`` (`float`, default: -1.0)
    This is contact matrix scaling factor for bars or other large social gatherings.
    Setting this to < 0 turns this transmission off.
* ``disease.nstrain`` (`integer`)
    The number of disease strains we're modeling.
* ``disease.p_trans`` (`list of float`, example: ``0.2 0.3``)
    These numbers are the probability of transmission given contact. There must be
    one entry for each disease strain.
* ``disease.p_asymp`` (`list of float`, example: ``0.4 0.4``)
    The fraction of cases that are asymptomatic. There must be
    one entry for each disease strain.
* ``disease.reduced_inf`` (`list of float`, example: ``0.75 0.75``)
    The relative infectiousness of asymptomatic individuals. There must be
    one entry for each disease strain.
* ``disease.reduced_inf`` (`float`, example: ``0.1``)
    The cross-strain reduction factor to the transmission probability.
* ``disease.vac_eff`` (`float`, example: ``0.4``)
    The vaccine efficacy - the probability of transmission will be multiplied by this factor

In addition to the ExaEpi inputs, there are also a number of runtime options that can be configured for AMReX itself. Please see <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__ for more information on these options.



