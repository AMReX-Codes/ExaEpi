/*! @file AirTravelFlow.H
    \brief #AirTravelFlow class definition
*/

#ifndef AIR_TRAFFIC_H_
#define AIR_TRAFFIC_H_

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>
#include "DemographicData.H"

#include <string>
typedef std::map<std::string, std::map<std::string, float> > airTravelType;

/*! \brief Structure containing information of traffic among major airport hubs*/
struct AirTravelFlow 
{
    /*! \brief Null constructor */
    AirTravelFlow (){}

    AirTravelFlow (const::std::string fname);

    void ReadAirTravelFlow (const std::string fname);
    void ReadAirports (const std::string fname, DemographicData& demo);
    void ComputeTravelProbs(DemographicData& demo);

    void Print () const;

    void CopyToDeviceAsync (const amrex::Vector<int>& h_vec, amrex::Gpu::DeviceVector<int>& d_vec);

    void CopyToHostAsync (const amrex::Gpu::DeviceVector<int>& d_vec, amrex::Vector<int>& h_vec);

    void CopyDataToDevice ();

    int N_hubs;                           /*!< number of airport hubs */
    amrex::Vector<int> FIPS_hubs;         /*!< FIPS code of each hub */
    amrex::Vector<int> Size_hubs;         /*!< Total passengers in each hub */
    amrex::Vector<int> num_org_passengers;       /*!< Passengers departing from the airport */
    amrex::Vector<int> num_dest_passengers;      /*!< Passengers arriving at the airport */
    std::map<int, std::string> airportMap;	  /*!< map from FIPS to airport code */
    std::map<std::string, std::vector<int> > airportRangeCounties; /*!< All counties served by an airport */
    std::map<std::string, int> airportRangePop; /*!< population of all counties served by an airport */
    std::map<std::string, int> originPax; 	/*!<number of origin passengers*/
    std::map<std::string, int> destPax; 	/*!<number of destination passengers*/
    std::map<int, float> travel_prob;  /*!< probability of a person in a given county to travel  */
    std::map<int, float> travel_to_prob;  /*!< probability of an air traveler that visit a given county */
    std::map<std::string/*org*/, std::map<std::string/*dest*/, float/*distribution*/> > travel_path_prob; /*!< probability of traveling between two airports */

    amrex::Gpu::DeviceVector<int> FIPS_hubs_d;         /*!< FIPS code of each hub (GPU) */
    amrex::Gpu::DeviceVector<int> Size_hubs_d;         /*!< Total passengers in each hub (GPU) */
    amrex::Gpu::DeviceVector<int> num_org_passengers_d;       /*!< Passengers departing from the airport (GPU) */
    amrex::Gpu::DeviceVector<int> num_dest_passengers_d;      /*!< Passengers arriving at the airport (GPU) */
};
#endif
