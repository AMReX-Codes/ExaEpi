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
* ``agent.initial_case_type`` (`string`)
    Either ``random`` or ``file``. If ``random``, ``agent.num_initial_cases`` must be set.
    If ``file``, ``agent.case_filename`` must be set. Must be provided if ``ic_type`` is ``"census"``.
* ``agent.case_filename`` (`string`)
    The path to the ``*.cases`` file containing the initial case data to use.
    Must be provided if ``initial_case_type`` is ``"file"``. Examples of these data files are provided
    in ``ExaEpi/data/CaseData``.
* ``agent.num_initial_cases`` (int)
    The number of initial cases to seed. Must be provided if ``initial_case_type`` is ``"random"``.
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
* ``agent.seed`` (`long integer`)
    Use this to specify the random seed to use for the run.
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
* ``disease.vac_eff`` (`float`, example: ``0.4``)
    The vaccine efficacy - the probability of transmission will be multiplied by this factor
* ``disease.incubation_length_mean`` (`float`, default: ``3.0``)
    Mean length of the incubation period in days. Before this, agents have no symptoms and are not infectious.
* ``disease.infectious_length_mean`` (`float`, default: ``6.0``)
    Mean length of the infectious period in days. This counter starts once the incubation phase is over. Before tihs, agents are symptomatic and can spread the disease.
* ``disease.symptomdev_length_mean`` (`float`, default: ``5.0``)
    Mean length of the time from exposure until symptoms develop in days. During the symptomatic-but-not-infectious stage agents  may self-withdraw, but they cannot spread the illness.
* ``disease.incubation_length_std`` (`float`, default: ``1.0``)
    Standard deviation of the incubation period in days.
* ``disease.infectious_length_std`` (`float`, default: ``1.0``)
    Standard deviation of the infectious period in days.
* ``disease.symptomdev_length_std`` (`float`, default: ``1.0``)
    Standard deviation of the time until symptom development in days.
* ``agents.size`` (`tuple of 2 integers`: e.g. ``(1, 1)``, default: ``(1, 1)``)
    This option is deprecated and will removed in a future version of ExaEpi. It controls
    the number of cells in the domain when running in `demo` mode. During actual usage,
    this number will be overridden and is irrelevant.
* ``agent.max_grid_size`` (`integer`, default: ``16``)
    This option sets the maximum grid size used for MPI domain decomposition. If set to
    ``16``, for example, the domain will be broken up into grids of `16^2` communities, and
    these grids will be assigned to different MPI ranks / GPUs.


In addition to the ExaEpi inputs, there are also a number of runtime options that can be configured for AMReX itself. Please see <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__ for more information on these options.



