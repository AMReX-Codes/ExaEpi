agent.ic_type = "census"
agent.census_filename = "../../data/CensusData/MA.dat"
agent.workerflow_filename = "../../data/CensusData/MA-wf.dat"

agent.initial_case_type = "file"
agent.case_filename = "../../data/CaseData/July4.cases"

#agent.initial_case_type = "random"
#agent.num_initial_cases = 5

agent.nsteps = 120
agent.plot_int = 10
agent.random_travel_int = 24

agent.aggregated_diag_int = -1
agent.aggregated_diag_prefix = "cases"

#agent.shelter_start = 7
#agent.shelter_length = 30
#agent.shelter_compliance = 0.85
#agent.mean_immune_time = 180
#agent.immune_time_spread = 60

contact.pSC  = 0.2
contact.pCO  = 1.45
contact.pNH  = 1.45
contact.pWO  = 0.5
contact.pFA  = 1.0
contact.pBAR = -1.

disease.nstrain = 1
disease.p_trans = 0.20
disease.p_asymp = 0.40
disease.reduced_inf = 0.75

disease.symptomdev_length_mean = 3.0
