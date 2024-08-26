/*! @file AirTravelFlow.cpp
    \brief Function implementations for AirTravelFlow class
*/

#include "AirTravelFlow.H"
#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <cmath>
#include <string>
#include <sstream>

using namespace amrex;

/*! \brief Constructor that initializes by reading in data from given file

    See AirTravelFlow::InitFromFile()
*/
AirTravelFlow::AirTravelFlow (const::std::string fname /*!< Filename to read case data from */)
{
}

void AirTravelFlow::ReadAirports(const std::string fname, DemographicData& demo){
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);
    
    std::string line;
    int airports=0;
    if(!is.eof()) { 
        getline(is, line); 
        std::istringstream lis(line);
        lis >> airports;
    }
    
    int FIPS = 0;
    std::string airportCode =  "";
    int i=0;
    while (airports>i++) {
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> FIPS >> airportCode;
        airportMap[FIPS]= airportCode;
	if(airportRangePop.find(airportCode)==airportRangePop.end()) airportRangePop[airportCode]= demo.CountyPop[FIPS];
	airportRangePop[airportCode]+= demo.CountyPop[FIPS];
    }
}

/*! \brief Read case data from a given file
	Description TBW
*/
void AirTravelFlow::ReadAirTravelFlow (const std::string fname /*!< Filename to read case data from */)
{
    BL_PROFILE("AirTravelFlow::InitFromFile");

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line;
    int airports=0;
    int connections=0;
    if(!is.eof()) {
        getline(is, line);
	std::istringstream lis(line);
        lis >> airports >> connections;
        //printf("AIRPORTS %d CONNECTION %d\n", airports, connections);
    }
    if(!connections) return;

    FIPS_hubs.resize(connections);
    Size_hubs.resize(connections);
    num_org_passengers.resize(connections);
    num_dest_passengers.resize(connections);

    FIPS_hubs_d.resize(connections);
    Size_hubs_d.resize(connections);
    num_org_passengers.resize(connections);
    num_dest_passengers.resize(connections);
    
    std::string dest = "";
    std::string org =  "";
    int pax = 0;
    int c=0;
    while (connections>c++) {
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> dest >> org >> pax;
	travel_path_prob[org][dest]= pax;

	if(originPax.find(org)== originPax.end()) originPax[org]= (int)  (pax/365);
	else originPax[org]+= (int)(pax/365);

	if(destPax.find(dest)== destPax.end()) destPax[dest]= (int)  (pax/365);
	else destPax[dest]+= (int)(pax/365);

    }
    //now we calculate the chance of traveling to a destination airport from an original one
    for(airTravelType::iterator it= travel_path_prob.begin(); it!= travel_path_prob.end(); it++)
    {
	std::string org= it->first;
	//find all paths from the origin airport
	int sum=0;
	for(std::map<std::string, float>::iterator it1= it->second.begin(); it1!= it->second.end(); it1++){
		std::string dest= it1->first;
		sum+=  travel_path_prob[org][dest];
	}
	
 	std::string prev_dest="";
	for(std::map<std::string, float>::iterator it1= it->second.begin(); it1!= it->second.end(); it1++){
		std::string dest= it1->first;
		if(it1 == it->second.begin())travel_path_prob[org][dest]= travel_path_prob[org][dest]/sum;
		else {
			travel_path_prob[org][dest]= travel_path_prob[org][prev_dest] + travel_path_prob[org][dest]/sum;
			prev_dest=dest;
		}
	}
    }
}

void AirTravelFlow::ComputeTravelProbs(DemographicData& demo){
    for(std::map<int, int>::iterator it= demo.CountyPop.begin(); it!= demo.CountyPop.end(); it++){
	travel_prob[it->first]= (float)originPax[airportMap[it->first]]/airportRangePop[airportMap[it->first]];
	travel_to_prob[it->first]= (float)destPax[airportMap[it->first]]/airportRangePop[airportMap[it->first]];
    }
    CopyDataToDevice();
    amrex::Gpu::streamSynchronize();
}

/*! \brief Prints case data to screen

    For each disease hub (#CaseData::N_hubs), print the FIPS code, current number of cases,
    and cumulative number of cases till date.
*/
void AirTravelFlow::Print () const {
}

/*! \brief Copy a vector from host to device */
void AirTravelFlow::CopyToDeviceAsync( const amrex::Vector<int>& h_vec,  /*!< Host vector */
                                  amrex::Gpu::DeviceVector<int>& d_vec  /*!< Device vector */) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

/*! \brief Copy a vector from device to host */
void AirTravelFlow::CopyToHostAsync( const amrex::Gpu::DeviceVector<int>& d_vec, /*!< Device vector */
                                amrex::Vector<int>& h_vec /*!< Host vector */) {
    h_vec.resize(0);
    h_vec.resize(d_vec.size());
    Gpu::copyAsync(Gpu::deviceToHost, d_vec.begin(), d_vec.end(), h_vec.begin());
}

/*! \brief Copy all member data from host to device */
void AirTravelFlow::CopyDataToDevice () {
    CopyToDeviceAsync(FIPS_hubs, FIPS_hubs_d);
    CopyToDeviceAsync(Size_hubs, Size_hubs_d);
}
