#include "AgentContainer.H"

using namespace amrex;

void AgentContainer::initAgents ()
{
    BL_PROFILE("AgentContainer::initAgents");

    // This gives us 1000 particles with random positions and unique ids,
    // all other components are zeroed out.
    Long num_agents = 1000;
    ULong iseed = 451;
    bool serialize = false;

    ParticleInitData pdata;
    pdata.real_array_data.fill(0.0);
    pdata.int_array_data.fill(0);
    this->InitRandom(num_agents, iseed, pdata, serialize);

    // Now we go back and make some of them positive, currently
    // no other attributes are actually used.
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);
        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const size_t np = soa.numParticles();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                if (amrex::Random(engine) < 0.05) {
                    status_ptr[i] = 1;
                }
            });
        }
    }
}

void AgentContainer::moveAgents ()
{
    BL_PROFILE("AgentContainer::moveAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                ParticleType& p = pstruct[i];

                p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[0]);
#if AMREX_SPACEDIM > 1
                p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                p.pos(2) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[2]);
#endif
            });
        }
    }
}

void AgentContainer::interactAgents ()
{
    BL_PROFILE("AgentContainer::interactAgents");

    // the dx used to bin the particles for spread is currently
    // twice the dx used to move the particles...

    IntVect bin_size = {AMREX_D_DECL(2, 2, 2)};
    for (int lev = 0; lev < numLevels(); ++lev)
    {
        const Geometry& geom = Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            amrex::DenseBins<ParticleType> bins;
            auto& ptile = ParticlesAt(lev, mfi);
            auto& aos   = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            const Box& box = mfi.validbox();

            int ntiles = numTilesInBox(box, true, bin_size);

            bins.build(np, pstruct_ptr, ntiles, GetParticleBin{plo, dxi, domain, bin_size, box});
            auto inds = bins.permutationPtr();
            auto offsets = bins.offsetsPtr();

            auto& soa   = ptile.GetStructOfArrays();
            auto d_ptr = soa.GetIntData(IntIdx::status).data();

            amrex::ParallelForRNG( bins.numBins(),
            [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
            {
                auto cell_start = offsets[i_cell];
                auto cell_stop  = offsets[i_cell+1];
                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if (d_ptr[pindex] == 1) {
                        for (unsigned int j = cell_start; j < cell_stop; ++j) {
                            if (i == j) { continue; }
                            auto pindex2 = inds[j];
                            if (amrex::Random(engine) < 0.5) {
                                d_ptr[pindex2] = 1;
                            }
                        }
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
}
