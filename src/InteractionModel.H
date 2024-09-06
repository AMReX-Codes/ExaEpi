/*! @file InteractionModel.H
    \brief Defines interaction model base class */

#ifndef _INTERACTION_MODEL_H_
#define _INTERACTION_MODEL_H_

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
    /*! \brief Name of models */
    namespace InteractionNames
    {
        const std::string generic = "generic";
        const std::string home = "home";
        const std::string work = "work";
        const std::string school = "school";
        const std::string nborhood = "neighborhood";
        const std::string transit = "transit";
        const std::string random = "random";
    }
}

/*! \brief Base class for defining interaction models
 *
 *  Contains things that are common to all interaction model classes.
 *  + The member data pointers are used to temporarily point to the SoA attributes of
 *    the agents for a given level and MFIter; for the various interaction models
 *    inheriting from this class, InteractionModel::getAgentAttribs() is called
 *    to set these pointers before using them.
 */
template <  typename AC /*!< agent container type */,
            typename ACT, /*!< agent container tile type */
            typename ACTD, /*!< agent container tile data type */
            typename A /*!< agent type */>
class InteractionModel
{
    public:

        /*! \brief null constructor */
        InteractionModel<AC,ACT,ACTD,A>() { }

        /*! \brief default destructor */
        virtual ~InteractionModel<AC,ACT,ACTD,A>() = default;

        /*! \brief Interact agents for a model */
        virtual void interactAgents(AC&, MultiFab&) = 0;

        /*! \brief Interact agents for a model */
        virtual void interactAgents(AC&, MultiFab&, AC&) = 0;

    protected:


        /*! \brief Get probability pointer from agent container */
        inline ParticleReal* getAgentProbPtr (  AC&           a_agents,  /*!< agent container */
                                                const int     a_lev,     /*!< level */
                                                const MFIter& a_mfi,     /*!< multifab iterator*/
                                                const int     a_d_idx    /*!< disease index */ )
        {
            BL_PROFILE("InteractionModel::getAgentProbPtr");
            auto& ptile = a_agents.ParticlesAt(a_lev, a_mfi);
            auto& soa   = ptile.GetStructOfArrays();
            int r_RT = RealIdx::nattribs;
            return soa.GetRealData(r_RT+r0(a_d_idx)+RealIdxDisease::prob).data();
        }

        /*! \brief Make particle bins for interactions at a given model */
        virtual void makeBins(  AC&                 a_agents,   /*!< agent container */
                                const IntVect&      a_bin_size, /*!< bin size */
                                const int           a_lev,      /*!< level */
                                const std::string&  a_mod       /*!< interaction model */ )
        {
            BL_PROFILE("InteractionModel::makeBins");
            const Geometry& geom = a_agents.Geom(a_lev);
            const auto dxi = geom.InvCellSizeArray();
            const auto plo = geom.ProbLoArray();
            const auto domain = geom.Domain();

            for(MFIter mfi = a_agents.MakeMFIter(a_lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
                auto bins_ptr = a_agents.getBins(pair_ind, a_mod);

                auto& ptile = a_agents.ParticlesAt(a_lev, mfi);
                auto& aos   = ptile.GetArrayOfStructs();
                const auto np = aos.numParticles();
                auto pstruct_ptr = aos().dataPtr();

                const Box& box = mfi.validbox();
                int ntiles = numTilesInBox(box, true, a_bin_size);

                auto binner = GetParticleBin{plo, dxi, domain, a_bin_size, box};
                bins_ptr->build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);

                AMREX_ALWAYS_ASSERT(np == bins_ptr->numItems());

                Gpu::synchronize();
            }
        }

    private:
};

#endif
