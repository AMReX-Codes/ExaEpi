#!=bin/bash

buildDIR=../../../build

cd $buildDIR/bin
AMDuProfPcm roofline -X -o ./exaEpi.csv -- ./agent ../../examples/inputs.census

AMDuProfModelling.py -i ./exaEpi.csv --operations float --plot roofline -o ./ --memspeed 3200 -a agent
AMDuProfModelling.py -i ./exaEpi.csv --operations int --plot roofline -o ./ --memspeed 3200 -a agent

