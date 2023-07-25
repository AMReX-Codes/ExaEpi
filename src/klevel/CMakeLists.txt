# List of source files
set(_sources
         main_klev.cpp
	 AgentContainerWrapper.H
	 AgentContainerWrapper.cpp
         ../AgentContainer.H
         ../AgentContainer.cpp
         ../CaseData.H
         ../CaseData.cpp
         ../DiseaseParm.H
         ../DiseaseParm.cpp
         ../DemographicData.H
         ../DemographicData.cpp
         ../Initialization.H
         ../Initialization.cpp
         ../IO.H
         ../IO.cpp
         ../Utils.H
         ../Utils.cpp)

# List of input files
set(_input_files )

setup_agent_klev(_sources _input_files)

unset( _sources )
unset( _input_files )
