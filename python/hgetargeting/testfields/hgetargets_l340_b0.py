import os
import numpy as np
from astropy.table import Table,vstack
from dlnpyutils import utils as dln,coords,plotting as pl
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt
import merge_2mass_vvv_fixed
#import merge_2mass_vvv_gaia_v7
from scipy.spatial import cKDTree
from scipy.stats import ncx2
from matplotlib.path import Path
from numpy.typing import ArrayLike, NDArray


# for LCO!!




def aperture_fraction(separation, fiber_radius=1.0, seeing_fwhm=1.2):
    """
    Fraction of a Gaussian PSF entering a circular fiber.

    separation, fiber_radius, and seeing_fwhm are in arcsec.
    """
    sigma = seeing_fwhm / 2.355

    x = (fiber_radius / sigma)**2
    nc = (np.asarray(separation) / sigma)**2

    return ncx2.cdf(x, df=2, nc=nc)

def neighbor_contamination(target_h, neighbor_h, separation,
                           fiber_radius=1.0, seeing_fwhm=1.2):

    neighbor_h = np.asarray(neighbor_h)
    separation = np.asarray(separation)

    target_fraction = aperture_fraction(
        0.0,
        fiber_radius=fiber_radius,
        seeing_fwhm=seeing_fwhm
    )

    neighbor_fraction = aperture_fraction(
        separation,
        fiber_radius=fiber_radius,
        seeing_fwhm=seeing_fwhm
    )

    flux_ratio = 10**(-0.4 * (neighbor_h - target_h))

    q = np.sum(
        flux_ratio * neighbor_fraction / target_fraction
    )

    fcontam = q / (1 + q)

    return fcontam

            
def brightneighbors(tab,rgb,magname='hmag',nneighbors=15,verbose=False):
    """ Flag targets with bright/close neighbors """

    # tab: all stars in this region
    # targetindex: index for the targets
    
    # Use KD-tree
    #X1 = np.asarray(X1, dtype=float)
    #X2 = np.asarray(X2, dtype=float)
    #N1, D = X1.shape
    #N2, D2 = X2.shape

    X1 = np.vstack((tab['ra'],tab['dec'])).T
    X2 = np.vstack((rgb['ra'],rgb['dec'])).T
    kdt = cKDTree(X1)

    dcr = 6.0/3600
    dist, ind = kdt.query(X2, k=nneighbors, distance_upper_bound=dcr)

    gdbrt, = np.where(np.isfinite(dist[:,1]))
    print(len(gdbrt),'of',len(targets),'targets have neighbors within 6"')
    nei = np.sum(np.isfinite(dist),axis=1)-1
    brightnei = np.zeros(len(rgb),bool)
    contam = np.zeros(len(rgb),float)
    for i in range(len(gdbrt)):
        index = gdbrt[i]
        if i % 5000 == 0: print(i)
        dist1 = dist[index,1:]*3600
        ind1 = ind[index,1:]
        gd1, = np.where(np.isfinite(dist1))

        # hit the nneighbor limit, do separate query
        if len(gd1)>=nneighbors-1:
            #print('hit neighbor limit')
            dist1, ind1 = kdt.query(X2[index:index+1,:], k=nneighbors+20, distance_upper_bound=dcr)
            dist1 = dist1.ravel()
            ind1 = ind1.ravel()
            gd1, = np.where(np.isfinite(dist1))
            
        dist1 = dist1[gd1]
        ind1 = ind1[gd1]
        hmag1 = tab[magname][ind1]
        minhmag1 = np.min(hmag1)
        rgbhmag = rgb[magname][index]

        fcontam = neighbor_contamination(
            target_h=rgbhmag,
            neighbor_h=hmag1,
            separation=dist1,
            fiber_radius=1.3/2,
            seeing_fwhm=1.2
        )

        contam[index] = fcontam

        if fcontam > 0.01:
            brightnei[index] = True

        if verbose:
            print(i,gdbrt[i],len(dist1),fcontam,fcontam>0.01)

        #if len(dist1)>=nneighbors-1:
        #    print('WARNING: star has hit the nneighbor limit')
        #    #import pdb; pdb.set_trace()
            
    return brightnei,contam


def select_spatially_uniform_indices(
    x: ArrayLike,
    y: ArrayLike,
    polygon_x: ArrayLike,
    polygon_y: ArrayLike,
    n_select: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.intp]:
    """Select points that approximately uniformly cover a 2D polygon.

    Uniform random positions are drawn inside the polygon. Each position is
    matched to the nearest eligible catalog point, without replacement.

    This reproduces the intent of the IDL ``uniform`` routine while replacing
    its ``ROI_CUT`` and ``SRCOR`` dependencies with Matplotlib and SciPy.

    Parameters
    ----------
    x, y
        Coordinates of the catalog points.
    polygon_x, polygon_y
        Coordinates of the polygon vertices. The final vertex need not repeat
        the first vertex.
    n_select
        Number of catalog points to select.
    rng
        NumPy random-number generator or seed. The default creates a new
        generator with an unpredictable seed.

    Returns
    -------
    numpy.ndarray
        Integer indices into the original ``x`` and ``y`` arrays.

    Notes
    -----
    The selected catalog points are not guaranteed to lie strictly inside the
    polygon if they lie numerically on its boundary; boundary points are
    treated as eligible. The distribution can only be as uniform as the
    density and geometry of the input catalog permit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    polygon_x = np.asarray(polygon_x, dtype=float)
    polygon_y = np.asarray(polygon_y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if polygon_x.ndim != 1 or polygon_y.ndim != 1:
        raise ValueError("polygon_x and polygon_y must be one-dimensional")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if polygon_x.size != polygon_y.size:
        raise ValueError("polygon_x and polygon_y must have the same length")
    if polygon_x.size < 3:
        raise ValueError("the polygon must have at least three vertices")
    if not isinstance(n_select, (int, np.integer)) or n_select <= 0:
        raise ValueError("n_select must be a positive integer")
    if n_select > x.size:
        raise ValueError("n_select cannot exceed the number of catalog points")
    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(polygon_x))
        and np.all(np.isfinite(polygon_y))
    ):
        raise ValueError("all coordinates must be finite")

    vertices = np.column_stack((polygon_x, polygon_y))
    closed_vertices = (
        vertices
        if np.array_equal(vertices[0], vertices[-1])
        else np.vstack((vertices, vertices[0]))
    )
    polygon = Path(closed_vertices, closed=True)
    points = np.column_stack((x, y))

    # A tiny positive radius includes points on the polygon boundary.
    inside = polygon.contains_points(points, radius=1e-12)
    eligible_indices = np.flatnonzero(inside)
    if eligible_indices.size < n_select:
        raise ValueError(
            f"only {eligible_indices.size} catalog points lie inside the "
            f"polygon, fewer than n_select={n_select}"
        )

    x_min, y_min = vertices.min(axis=0)
    x_max, y_max = vertices.max(axis=0)
    if x_min == x_max or y_min == y_max:
        raise ValueError("the polygon must have nonzero width and height")

    generator = np.random.default_rng(rng)

    # Draw targets in batches to avoid one-at-a-time rejection sampling.
    targets: list[NDArray[np.float64]] = []
    n_targets = 0
    while n_targets < n_select:
        batch_size = max(256, 2 * (n_select - n_targets))
        candidates = generator.uniform(
            low=(x_min, y_min), high=(x_max, y_max), size=(batch_size, 2)
        )
        accepted = candidates[polygon.contains_points(candidates, radius=1e-12)]
        if accepted.size:
            targets.append(accepted)
            n_targets += len(accepted)
    target_points = np.concatenate(targets, axis=0)[:n_select]

    eligible_points = points[eligible_indices]
    tree = cKDTree(eligible_points)
    used = np.zeros(eligible_indices.size, dtype=bool)
    selected = np.empty(n_select, dtype=np.intp)

    for i, target in enumerate(target_points):
        # Usually k=1 is sufficient. Increase k only when a target's nearest
        # neighbor was already assigned to an earlier target.
        k = 1
        while True:
            _, neighbors = tree.query(target, k=k)
            neighbors = np.atleast_1d(neighbors)
            available = neighbors[~used[neighbors]]
            if available.size:
                chosen = int(available[0])
                break
            k = min(2 * k, eligible_indices.size)

        used[chosen] = True
        selected[i] = eligible_indices[chosen]

    return selected




def targets():

    tag = 'hge_l340_b0.0_rad1.05'
    
    tmass = Table.read('hge_l340_b0.0_rad1.05_2mass.fits.gz')
    tmass['decl'].name = 'dec'
    tmass['glon'].name = 'l'
    tmass['glat'].name = 'b'
    # twomass_psc glon/glat are low precision
    coo = SkyCoord(tmass['ra'],tmass['dec'],unit='degree',frame='icrs')
    tmass['l'] = coo.galactic.l.degree
    tmass['b'] = coo.galactic.b.degree
    tmass['j_m'].name = 'jmag'
    tmass['h_m'].name = 'hmag'
    tmass['k_m'].name = 'ksmag'
    # photometric quality cuts
    bad_phot = set('UEFX')
    good_2mass = np.array([
        not any(flag in bad_phot for flag in str(x))
        for x in tmass['ph_qual']
    ])
    tmass['good_2mass'] = good_2mass

    # VIRAC2
    vvv = Table.read('hge_l340_b0.0_rad1.05_virac2_lite.fits.gz')
    vvv['de'].name = 'dec'
    vvv['phot_j_mean_mag'].name = 'jmag'
    vvv['phot_j_std_mag'].name = 'e_jmag'
    vvv['phot_h_mean_mag'].name = 'hmag'
    vvv['phot_h_std_mag'].name = 'e_hmag'
    vvv['phot_ks_mean_mag'].name = 'ksmag'
    vvv['phot_ks_std_mag'].name = 'e_ksmag'
    coo = SkyCoord(vvv['ra'],vvv['dec'],unit='degree',frame='icrs')
    vvv['l'] = coo.galactic.l.degree
    vvv['b'] = coo.galactic.b.degree
    # photometric quality cuts
    good_vvv = (np.isfinite(vvv['jmag']) & np.isfinite(vvv['hmag']) & np.isfinite(vvv['ksmag']) &
                (vvv['e_jmag']<=0.1) & (vvv['e_hmag']<=0.1) & (vvv['e_ksmag']<=1.0))
    vvv['good_vvv'] = good_vvv
    

    # APOGLIMPSE
    glimpse = Table.read('hge_l340_b0.0_rad1.05_apoglimpse.fits.gz')



    def values(col, fill=np.nan):
        return np.asarray(np.ma.filled(col, fill))

    mag45 = values(glimpse['mag4_5'])
    err45 = values(glimpse['d4_5m'])
    sn45  = values(glimpse['sn_4_5'])
    m45   = values(glimpse['m4_5'], fill=0)
    n45   = values(glimpse['n4_5'], fill=0)
    sqf45 = values(glimpse['sqf_4_5'], fill=-1).astype(np.int64)
    csf   = values(glimpse['csf'], fill=99)

    detfrac45 = np.zeros(len(glimpse), dtype=float)

    use = n45 > 0
    detfrac45[use] = m45[use] / n45[use]

    bad_bits = [
        1,   # poor pixels in dark-current calibration
        2,   # questionable flat field
        3,   # latent image
        8,   # hot/dead/unacceptable pixel
        13,  # confusion in in-band merge
        14,  # confusion in cross-band merge
        19,  # predicted saturation
        20,  # saturated-star wing
        21,  # in-band source lumping
        22,  # cross-band source lumping
        30,  # near frame edge
    ]

    bad_sqf_mask = sum(1 << (bit - 1) for bit in bad_bits)

    good_sqf45 = (
        (sqf45 >= 0) &
        ((sqf45 & bad_sqf_mask) == 0)
    )

    good_45 = (
        np.isfinite(mag45) &
        np.isfinite(err45) &
        np.isfinite(sn45) &
        (sn45 >= 10) &
        (err45 < 0.10) &
        (m45 >= 2) &
        (detfrac45 >= 0.60) &
        good_sqf45 &
        (csf <= 2)
    )

    glimpse['good_45'] = good_45




    # ---- On-sky spatial density plots ----
    
    phi = np.linspace(0,2*np.pi,100)
    xr = [341.1,338.90]
    yr = [-1.0,1.0]
    
    # 2MASS
    o=pl.hist2d(tmass['l'],tmass['b'],xr=xr,yr=yr,log=True,xtitle='Galactic Longitude (deg)',
                ytitle='Galactic Latitude (deg)',title='2MASS')
    plt.plot(1.05*np.cos(phi)+340,1.05*np.sin(phi),'k')
    plt.savefig(tag+'_tmass_skydensity.png',bbox_inches='tight')

    # VIRAC2
    o=pl.hist2d(vvv['l'],vvv['b'],xr=xr,yr=yr,log=True,xtitle='Galactic Longitude (deg)',
                ytitle='Galactic Latitude (deg)',title='VIRAC2')
    plt.plot(1.05*np.cos(phi)+340,1.05*np.sin(phi),'k')
    plt.savefig(tag+'_virac2_skydensity.png',bbox_inches='tight')

    # GLIMPSE
    o=pl.hist2d(glimpse['l'],glimpse['b'],xr=xr,yr=yr,log=True,xtitle='Galactic Longitude (deg)',
                ytitle='Galactic Latitude (deg)',title='GLIMPSE')
    plt.plot(1.05*np.cos(phi)+340,1.05*np.sin(phi),'k')
    plt.savefig(tag+'_apoglimpse_skydensity.png',bbox_inches='tight')


    
    
    # ---- Color Magnitude Diagrams ----

    # 2MASS
    o=pl.hist2d(tmass['jmag']-tmass['ksmag'],tmass['hmag'],xr=[-1,10],yr=[20,5],log=True,
                xtitle='J-Ks',ytitle='H',title='2MASS')
    plt.savefig(tag+'_tmass_cmd.png',bbox_inches='tight')

    # VIRAC2
    o=pl.hist2d(vvv['jmag']-vvv['ksmag'],vvv['hmag'],xr=[-1,10],yr=[20,5],log=True,
                xtitle='J-Ks',ytitle='H',title='VIRAC2')
    plt.savefig(tag+'_virac2_cmd.png',bbox_inches='tight')

    # GLIMPSE
    o=pl.hist2d(glimpse['mag3_6']-glimpse['mag4_5'],glimpse['mag4_5'],xr=[-1,3],yr=[16,5],log=True,
                xtitle='[3.6]-[4.5]',ytitle='[4.5]',title='APOGLIMPSE')
    plt.savefig(tag+'_apoglimpse_cmd.png',bbox_inches='tight')


    # ---- Photometric transformation of VIRAC2 to 2MASS system ----
    # determine zero-point offset using the data
    

    # ---- Merge 2MASS + VIRAC2 ----

    nir = merge_2mass_vvv_fixed.merge(tmass,vvv)

    good_2mass = np.ma.filled(nir['good_2mass'], False).astype(bool)
    good_vvv = np.ma.filled(nir['good_vvv'], False).astype(bool)
    nir['good_nir'] = good_2mass | good_vvv

    o=pl.hist2d(nir['jmag_2mass']-nir['kmag_2mass'],nir['hmag_2mass'],xr=[-1,10],yr=[20,5],log=True,
                xtitle='J-Ks',ytitle='H',title='Merged 2MASS+VIRAC2 CMD')
    plt.savefig(tag+'_tmassvirac2_cmd.png',bbox_inches='tight')



    
    # ---- Crossmatch to GLIMPSE ----
    
    ind1,ind2,dist = coords.xmatch(glimpse['ra'],glimpse['dec'],nir['ra'],nir['dec'],1.0,unique=True)
    print(len(ind1),'matches of APOGLIMPSE ({:d}) to merged NIR catalog ({:d})'.format(len(glimpse),len(nir)))
    nir['glimpse_match'] = False
    nir['glimpse_mag4_5'] = np.nan
    nir['glimpse_mag3_6'] = np.nan
    nir['glimpse_good'] = False
    nir['glimpse_match'][ind2] = True
    nir['glimpse_mag3_6'][ind2] = glimpse['mag3_6'][ind1]
    nir['glimpse_mag4_5'][ind2] = glimpse['mag4_5'][ind1]
    nir['glimpse_good'][ind2] = glimpse['good_45'][ind1]

    

    # ---- RJCE dereddening ----
    # Use Zasowski+2009 for the particular longitude
    # use the color excess ratios
    
    nir['ak'] = np.nan
    nir['ah'] = np.nan
    nir['ejk'] = np.nan
    nir['jk0'] = np.nan
    nir['h0'] = np.nan

    # Zasowski+2009 guided RJCE quation
    # eh45 = (hmag - mag45) - 0.08
    # aks = 0.928 * eh45
    # ah = 1.55 * aks
    # ejk = 1.647 * aks
    # jk0 = (jmag - kmag) - ejk
    # h0 = hmag - ah
    gd, = np.where((nir['hmag_2mass'] < 50) & np.isfinite(nir['glimpse_mag4_5']))
    nir['ak'][gd] = 0.928*(nir['hmag_2mass'][gd] - nir['glimpse_mag4_5'][gd] - 0.08)
    nir['ah'][gd] = 1.55*nir['ak'][gd]
    nir['ejk'][gd] = 1.647*nir['ak'][gd]
    nir['jk0'][gd] = nir['jmag_2mass'][gd]-nir['kmag_2mass'][gd]-nir['ejk'][gd]
    nir['h0'][gd] = nir['hmag_2mass'][gd]-nir['ah'][gd]

    o=pl.hist2d(nir['jk0'],nir['h0'],xr=[-1,2.0],yr=[16,4],log=True,
                xtitle=r'(J-Ks)$_0$',ytitle=r'H$_0$',title='RJCE Dereddened CMD')
    plt.savefig(tag+'_tmassvirac2_deredcmd.png',bbox_inches='tight')
    
    
    # ---- Photometric quality cuts ----
    
    good, = np.where(nir['good_nir'] & nir['glimpse_good'])
    nir_good = nir[good]

    o=pl.hist2d(nir_good['jk0'],nir_good['h0'],xr=[-1,2.0],yr=[16,4],log=True,
                xtitle=r'(J-Ks)$_0$',ytitle=r'H$_0$',title='RJCE Dereddened CMD after quality cuts')
    plt.savefig(tag+'_tmassvirac2_deredcmd_qacuts.png',bbox_inches='tight')
    

    
    
    # ---- RGB selection ----
    jkcut = [0.80,1.4,1.4,0.80,0.80]
    hcut = [11.8+1.0,9.8+1.0,5.4-1,7.8-1,11.8+1.0]
    ind,cutind = dln.roi_cut(jkcut,hcut,nir_good['jk0'],nir_good['h0'])
    alltargets = nir_good[cutind]
    print(len(alltargets),'RGB targets')

    
    # Dereddened CMD of targets
    o=pl.hist2d(nir_good['jk0'],nir_good['h0'],xr=[-1,2.0],yr=[16,4],log=True,
                xtitle=r'(J-Ks)$_0$',ytitle=r'H$_0$',title='RJCE Dreddened CMD - RGB target selection')
    plt.plot(jkcut,hcut,c='r')
    plt.scatter(alltargets['jk0'],alltargets['h0'],s=2,c='orange')
    plt.text(0.95, 0.05, '{} targets'.format(len(targets)),
             transform=plt.gca().transAxes,ha='right',va='top')    
    plt.savefig(tag+'_alltargets_deredcmd.png',bbox_inches='tight')



    # ---- Close neighbors ----
    # fibers are 1.3" in diameter
    # use Gaia and nir as master catalogs

    brt_nir,fcontam_nir = brightneighbors(nir,alltargets,'hmag')
    alltargets['brtnei'] = brt_nir
    alltargets['brtnei_fcontam'] = fcontam_nir
    print('{:d} targets with close neighbors ({:.2f}%)'.format(np.sum(brt_nir),np.sum(brt_nir)/len(alltargets)*100))


    o=pl.scatter(alltargets['jmag_2mass']-alltargets['kmag_2mass'],alltargets['hmag_2mass'],size=2,xr=[-1,10],yr=[18,5],
                 xtitle='J-Ks',ytitle='H',title='All Targets - Bright neighbors')
    plt.scatter(alltargets['jmag_2mass'][brt_nir]-alltargets['kmag_2mass'][brt_nir],
                alltargets['hmag_2mass'][brt_nir],s=2,c='r',label='Bright Neighbors ({:d})'.format(np.sum(brt_nir)))
    plt.legend()
    plt.savefig(tag+'_targets_cmd_brtnei.png',bbox_inches='tight')


    
    # ---- Apply Final Selection ----
    
    # Apply hard magnitude limits
    hmin = 11.0
    hmax = 17.0
    gd, = np.where((alltargets['hmag_2mass'] >= hmin) & (alltargets['hmag_2mass'] <= hmax))
    targets = alltargets[gd]
    print(len(gd),'targets after applying {:.2f} <= H <= {:.2f}'.format(hmin,hmax))
    
    # Apply close neighbor limits
    fcontam_thresh = 0.01
    gd, = np.where(targets['brtnei_fcontam'] < fcontam_thresh)
    targets = targets[gd]
    print(len(gd),'targets after applying brtnei_fcontam < {:.3f}'.format(fcontam_thresh))
    
    # Use uniform CMD selection
    print('Applying uniform CMD selection prioritization')
    uniform_order = select_spatially_uniform_indices(targets['jk0'],targets['h0'],
                                                     jkcut,hcut,n_select=len(targets),rng=42)
    priority = np.empty(len(targets), dtype=int)
    priority[uniform_order] = np.arange(len(targets))
    targets["priority"] = priority

    
    # Final CMD of targets
    o=pl.hist2d(nir_good['jmag_2mass']-nir_good['kmag_2mass'],nir_good['hmag_2mass'],
                xr=[-1,10],yr=[18,5],log=True,xtitle='J-Ks',ytitle='H',title='Final targets')
    plt.scatter(targets['jmag_2mass']-targets['kmag_2mass'],targets['hmag_2mass'],s=2,c='r')
    plt.savefig(tag+'_targets_cmd.png',bbox_inches='tight')

    # Final CMD of targets
    o=pl.hist2d(nir_good['jk0'],nir_good['h0'],xr=[-1,2],yr=[16,4],log=True,xtitle=r'(J-Ks)$_0$',ytitle=r'H$_0$',
                title='Dereddened CMD - Final targets')
    plt.scatter(targets['jk0'],targets['h0'],s=2,c='r')
    plt.savefig(tag+'_targets_deredcmd.png',bbox_inches='tight')

    
    # Save final targets to file
    targets.write(tag+'_targets.fits',overwrite=True)

    # ----- Separate bright and faint designs ----



    

    # Make HTML page for the diagnostic plots
    

    import pdb; pdb.set_trace()
