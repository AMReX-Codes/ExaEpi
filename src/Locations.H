/*! @file Locations.H
 *  \brief Defines interaction for various locations
 */

#ifndef _LOCATIONS_H_
#define _LOCATIONS_H_

#include <map>
#include <string>

#include <AMReX_MultiFab.H>

namespace ExaEpi
{
    /*! \brief Name of locations */
    namespace LocationNames
    {
        const std::string generic = "generic";
        const std::string home = "home";
        const std::string work = "work";
        const std::string school = "school";
        const std::string transit = "transit";
    };
};

/*! \brief Base class for defining locations and agent interactions at them */
template <typename ACType /*!< agent container type */, typename AType /*!< agent type */>
class InteractionLocation
{
    public:

        /*! \brief null constructor */
        InteractionLocation<ACType,AType>() { }

        /*! \brief default destructor */
        virtual ~InteractionLocation<ACType,AType>() = default;

        /*! \brief Interact agents at a location */
        virtual void interactAgents(ACType&, amrex::MultiFab&) = 0;

    protected:

    private:
};

#endif
