
import numpy as np
import healpy as hp


def simulate_sky(Cl_gg, Cl_vg, Cl_vv, settings, seed):

    nbins = settings.nbins
    ell_max = settings.lmax
    NSIDE = settings.nside
    NPIX = hp.nside2npix(NSIDE)

    # -------------------------------------------
    # 1. Construct 16 x 16 covariance
    # -------------------------------------------
    C = np.zeros((2*nbins, 2*nbins, ell_max+1))
    C[:nbins, :nbins] = Cl_gg
    C[nbins:, nbins:] = Cl_vv
    C[nbins:, :nbins] = Cl_vg
    C[:nbins, nbins:] = Cl_vg.transpose(1, 0, 2)

    # -------------------------------------------
    # 2. HEALPix synalm ordering
    # -------------------------------------------
    cls_hp = []
    for i in range(2*nbins):
        cls_hp.append(C[i, i])
    for d in range(1, 2*nbins):
        for i in range(2*nbins-d):
            j = i + d
            cls_hp.append(C[i, j])

    # -------------------------------------------
    # 3. Generate correlated Gaussian alms
    # -------------------------------------------
    np.random.seed(seed)

    alms = hp.synalm(
        cls_hp,
        lmax=ell_max,
        new=True
    )

    # -------------------------------------------
    # 4. Generate maps
    # -------------------------------------------
    maps_g = np.zeros((nbins, NPIX))
    maps_v = np.zeros((nbins, NPIX))

    for i in range(nbins):
        maps_g[i] = hp.alm2map(
            alms[i], nside=NSIDE, lmax=ell_max
        )
        maps_v[i] = hp.alm2map(
            alms[nbins+i], nside=NSIDE, lmax=ell_max
        )

    # -------------------------------------------
    # 5. Calculate realised Cl
    # -------------------------------------------
    Cl_gg_real = np.zeros((nbins, nbins, ell_max+1))
    Cl_vg_real = np.zeros((nbins, nbins, ell_max+1))
    Cl_vv_real = np.zeros((nbins, nbins, ell_max+1))

    for i in range(nbins):
        for j in range(nbins):
            Cl_gg_real[i, j] = hp.alm2cl(
                alms[i], alms[j]
            )
            Cl_vg_real[i, j] = hp.alm2cl(
                alms[nbins+i], alms[j]
            )
            Cl_vv_real[i, j] = hp.alm2cl(
                alms[nbins+i], alms[nbins+j]
            )

    # -------------------------------------------
    # 6. Return one simulated realisation
    # -------------------------------------------
    return {
        "alms": alms,
        "maps_g": maps_g,
        "maps_v": maps_v,
        "Cl_gg": Cl_gg_real,
        "Cl_vg": Cl_vg_real,
        "Cl_vv": Cl_vv_real
    }
