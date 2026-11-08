import numpy as np
import healpy as hp

def extract_features(sim, settings):

    if settings.input_type == "cl":
        Cl_gg = sim["Cl_gg"][:, :, settings.ell_min:]
        Cl_vg = sim["Cl_vg"][:, :, settings.ell_min:]
        Cl_vv = sim["Cl_vv"][:, :, settings.ell_min:]

        spectra = {
            "gg": Cl_gg,
            "vg": Cl_vg,
            "vv": Cl_vv
        }
        x = np.concatenate([
            spectra[name].ravel()
            for name in settings.spectra
        ])

    elif settings.input_type == "map":
        maps = np.concatenate([
            sim["maps_g"],
            sim["maps_v"]
        ])
        x = maps.ravel().astype(np.float32)

    elif settings.input_type == "alm":
        maps = np.concatenate([
            sim["maps_g"],
            sim["maps_v"]
        ])

        ell_idx, _ = hp.Alm.getlm(settings.lmax)
        keep_alm = ell_idx >= settings.ell_min

        alm_all = []

        for i in range(2 * settings.nbins):
            alm = hp.map2alm(
                maps[i],
                lmax=settings.lmax,
                iter=3
            )
            alm = alm[keep_alm]

            alm_all.append(
                np.concatenate([
                    alm.real,
                    alm.imag
                ])
            )
        x = np.concatenate(alm_all).astype(np.float32)
        
    else:
        raise ValueError(
            f"Unknown input type: {settings.input_type}"
        )
    return x