# List of source files
set(_sources
         main.cpp
         AgentContainer.H
         AgentContainer.cpp
         DemographicData.H
         DemographicData.cpp
         Initialization.H
         Initialization.cpp
         IO.H
         IO.cpp
         Utils.H
         Utils.cpp)

# List of input files
set(_input_files )

setup_agent(_sources _input_files)

unset( _sources )
unset( _input_files )
