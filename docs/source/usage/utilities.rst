Output Format
=============

ExaEpi writes data on two levels: individual data and community data. 

Individual data fields:
-----------------------
Individual datasets have more fields written on day 0 that are constant across the simulation. Data is either written as an ``int`` or a ``real``.

- Day 0 only (constant)
  
  - Int data:
    
    - ``age_group``: Ranges from 0 to 4, indicating age group of agent. Age ranges are 0-5, 5-17, 18-29, 30-64, and 65+, respectively.

    - ``family``: ID of family of agent. 

    - ``home_i``: x-coordinate of home location on AMReX grid. Can be used to match particle to home community.

    - ``home_j``: y-coordinate of home location on AMReX grid. Can be used to match particle to home community.

    - ``work_i``: x-coordinate of work location on AMReX grid. Can be used to match particle to work community.

    - ``work_j``: y-coordinate of work location on AMReX grid. Can be used to match particle to work community.

    - ``nborhood``: Home neighborhood.

    - ``school``: Ranges from -1 to 5, indicating type of school. -1 indicates not of school age, 0 indicates of school age but not in school, 1 indicates high school, 2 indicates middle school, and 3-5 indicate different elementary schools.

    - ``work_nborhood``: Work neighborhood.

    - ``strain``: Marks which strain of disease (used for multiple diseases).

  - Real data:

    - ``incubation_period``: (currently misleadingly named) Latent period length, i.e. time between exposure and infectiousness.

    - ``infectious_period``: Infections period length, i.e. time during which agent can spread disease

    - ``symptomdev_period``: Symptom development period length (known as incubation period in some contexts), i.e. time between exposure and symptoms appearing.

- Every day (time-varying)

  - Int data

    - ``withdrawn``: Marks whether the agent is withdrawn. 0 for not withdrawn, 1 for withdrawn.

    - ``status``: Ranges from 0 to 4, indicating disease status. Corresponds to never infected, infected, immune, susceptible, and dead, respectively.

    - ``symptomatic``: Ranges from 0 to 2, indicating symptomaticity. Corresponds to presymptomatic (not yet symptomatic but will be), symptomatic, and a symptomatic.

  - Real data

    - ``treatment_timer``: Time that agent has been in hospital.

    - ``disease_counter``: Time that agent has been infected.

    - ``infection_prob``: Probability that agent is infected at end of day.

Community data fields:
----------------------
All data is written as a real, although all data should be integers. If any community does not exist, its ``unit``, ``FIPS``, ``Tract``, and ``comm`` values will be -1, while all agent counts will be 0.

- ``total``: Total population in community.

- ``never_infected``: Number of never-infected agents in community.

- ``infected``: Number of infected agents in community.

- ``immune``: Number of immune (recently recovered from disease) agents in community.

- ``susceptible``: Number of susceptible (recovered from disease long enough ago to be able to be infected again) agents in community.

- ``unit``: Unit at cell.

- ``FIPS``: FIPS code (county code) of community.

- ``Tract``: Census tract of community.

- ``comm``: Community ID (unique).
