/*! @file Locations.H
 *  \brief Defines interaction for various locations
 */

#ifndef _LOCATIONS_H_
#define _LOCATIONS_H_

#include <map>
#include <string>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include "AgentDefinitions.H"

using namespace amrex;

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
    }
}

/*! \brief Base class for defining locations and agent interactions at them
 *
 *  Contains things that are common to all interaction location classes.
 *  + The member data pointers are used to temporarily point to the SoA attributes of
 *    the agents for a given level and MFIter; for the various interaction locations
 *    inheriting from this class, InteractionLocation::getAgentAttribs() is called
 *    to set these pointers before using them.
 */
template <typename ACType /*!< agent container type */, typename AType /*!< agent type */>
class InteractionLocation
{
    public:

        /*! \brief null constructor */
        InteractionLocation<ACType,AType>() { }

        /*! \brief default destructor */
        virtual ~InteractionLocation<ACType,AType>() = default;

        /*! \brief Interact agents at a location */
        virtual void interactAgents(ACType&, MultiFab&) = 0;

    protected:

        const int* m_home_i_ptr; /*!< i-index of home location */
        const int* m_home_j_ptr; /*!< j-index of home location */
        const int* m_work_i_ptr; /*!< i-index of work location */
        const int* m_work_j_ptr; /*!< j-index of work location */

        const int* m_status_ptr; /*!< infection status */
        const int* m_age_group_ptr; /*!< age group */
        const int* m_family_ptr; /*!< family */
        const int* m_nborhood_ptr; /*!< neighborhood */
        const int* m_school_ptr; /*!< school */
        const int* m_workgroup_ptr; /*!< workgroup */
        const int* m_withdrawn_ptr; /*!< withdrawn/quarrantining */
        const int* m_strain_ptr; /*!< disease strain */

        const ParticleReal* m_counter_ptr; /*!< day counter since infection */
        const ParticleReal* m_timer_ptr; /*!< timer since onset of disease */

        ParticleReal* m_prob_ptr; /*!< infection probability */

        /*! \brief Get attribute pointers from agent container */
        virtual void getAgentAttribs ( ACType&       a_agents,  /*!< agent container */
                                       const int     a_lev,     /*!< level */
                                       const MFIter& a_mfi      /*!< multifab iterator*/ )
        {
            BL_PROFILE("InteractionLocation::getAgentAttribs");

            auto& ptile = a_agents.ParticlesAt(a_lev, a_mfi);
            auto& soa   = ptile.GetStructOfArrays();

            m_work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            m_work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

            m_status_ptr = soa.GetIntData(IntIdx::status).data();
            m_age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            m_family_ptr = soa.GetIntData(IntIdx::family).data();
            m_nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
            m_school_ptr = soa.GetIntData(IntIdx::school).data();
            m_workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            m_withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            m_counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();

            //m_strain_ptr = soa.GetIntData(IntIdx::strain).data();
            //m_timer_ptr = soa.GetRealData(RealIdx::timer).data();

            m_prob_ptr = soa.GetRealData(RealIdx::prob).data();
        }

        /*! \brief Make particle bins for interactions at a given location */
        virtual void makeBins(  ACType&             a_agents,   /*!< agent container */
                                const IntVect&      a_bin_size, /*!< bin size */
                                const int           a_lev,      /*!< level */
                                const std::string&  a_loc       /*!< location */ )
        {
            BL_PROFILE("InteractionLocation::makeBins");
            const Geometry& geom = a_agents.Geom(a_lev);
            const auto dxi = geom.InvCellSizeArray();
            const auto plo = geom.ProbLoArray();
            const auto domain = geom.Domain();

            for(MFIter mfi = a_agents.MakeMFIter(a_lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
                auto bins_ptr = a_agents.getBins(pair_ind, a_loc);

                if (bins_ptr->numBins() < 0) {

                    auto& ptile = a_agents.ParticlesAt(a_lev, mfi);
                    auto& aos   = ptile.GetArrayOfStructs();
                    const auto np = aos.numParticles();
                    auto pstruct_ptr = aos().dataPtr();

                    const Box& box = mfi.validbox();
                    int ntiles = numTilesInBox(box, true, a_bin_size);

                    auto binner = GetParticleBin{plo, dxi, domain, a_bin_size, box};
                    bins_ptr->build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);

                    AMREX_ALWAYS_ASSERT(np == bins_ptr->numItems());
                }

                Gpu::synchronize();
            }
        }

    private:
};

#endif
