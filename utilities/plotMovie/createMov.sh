#!/bin/bash
module load python

for i in "$@"; do
  case $i in
    -c=*|--censusMap=*)
      CENSUS="${i#*=}"
      shift # past argument=value
      ;;
    -i=*|--infectionCases=*)
      INFECTION="${i#*=}"
      shift # past argument=value
      ;;
    -d=*|--casesMetric=*) #CUML/DAILY
      INPUT_METRIC="${i#*=}"
      shift # past argument=value
      ;;
    -m=*|--plotMetric=*) #CUML/DAILY
      OUTPUT_METRIC="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "PATH TO INFECTION CASES  = ${INFECTION}"
echo "PATH TO CENSUS MAP    = ${CENSUS}"
echo "CASES METRIC  = ${INPUT_METRIC}"
echo "METRIC TO PLOT = ${OUTPUT_METRIC}"

#plot everyday's cases on a cencus map
python plotCases.py ${INFECTION} ${CENSUS} ${INPUT_METRIC} ${OUTPUT_METRIC}

#create movie
ffmpeg -framerate 1/4 -i ca%d.png -c:v lib264 -r 30 -pix_fmt yuv420p movie.mp4

#remove temporary files
rm *.png
