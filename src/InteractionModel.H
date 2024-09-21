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
    AMREX_ENUM(InteractionNames, home, work, school, home_nborhood, work_nborhood, transit, random);
}

/*! \brief Base class for defining interaction models
 *
 *  Contains things that are common to all interaction model classes.
 *  + The member data pointers are used to temporarily point to the SoA attributes of
 *    the agents for a given level and MFIter; for the various interaction models
 *    inheriting from this class, InteractionModel::getAgentAttribs() is called
 *    to set these pointers before using them.
 */
template <typename PCType /*!< agent container type */,
          typename PTDType /*!< agent container tile data type */,
          typename PType /*!< agent type */>
class InteractionModel
{
    public:

        /*! \brief null constructor */
        InteractionModel<PCType, PTDType, PType> () {}

        /*! \brief default destructor */
        virtual ~InteractionModel<PCType, PTDType, PType> () = default;

        /*! \brief Interact agents for a model */
        virtual void interactAgents(PCType&, MultiFab&) = 0;

};

template <typename PTDType>
struct GetCommunityIndex
{
        GetCommunityIndex (const Geometry &geom, const Box &_valid_box) :  valid_box(_valid_box) {
            dxi = geom.InvCellSizeArray();
            plo = geom.ProbLoArray();
            domain = geom.Domain();
            bin_size = {AMREX_D_DECL(1, 1, 1)};
        }

        AMREX_GPU_HOST_DEVICE
        int operator() (const PTDType& ptd, int i) const noexcept {
            Box tbx;
            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
            return getTileIndex(iv, valid_box, true, bin_size, tbx);
        }

        AMREX_GPU_HOST_DEVICE
        int max() {
            return numTilesInBox(valid_box, true, bin_size);
        }


    private:

        GpuArray<Real,AMREX_SPACEDIM> plo;
        GpuArray<Real,AMREX_SPACEDIM> dxi;
        Box domain;
        IntVect bin_size;
        Box valid_box;
};



#endif
