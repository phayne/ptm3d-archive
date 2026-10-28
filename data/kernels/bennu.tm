KPL/MK

   PTM3D meta-kernel: asteroid (101955) Bennu, generic-body SPICE path.
   Used by docs/tutorials/notebooks/04_asteroid_spice.ipynb and
   examples/quickstart_asteroid_bennu.yaml.

   Paths are RELATIVE to the repository root -- run ptm_thermal_sim from the
   repo root (or edit PATH_VALUES to an absolute path).

   This file lives in data/kernels/ (a bare filename, like moon.tm/europa.tm/
   mercury.tm) so that generic_body.spice_metakernel resolves it with no
   further configuration. An absolute path, or a path relative to the working
   directory, also resolves -- see docs/configuration.md, "How
   generic_body.spice_metakernel is resolved". (Before the generic-body metakernel
   fix, only the bare-filename form worked; older notes saying so are stale.)

   BENNU'S EPHEMERIS IS NOT IN GIT. Like de440s.bsp it is fetched:

       python3 scripts/fetch_kernels.py --bodies bennu

   (the default, --bodies auto, also fetches it, because this metakernel
   ships). Until you do, a Bennu run stops with SPICE(NOSUCHFILE) naming the
   missing file -- measured, not assumed. It does not quietly fall back.

   WHICH EPHEMERIS, AND WHY THIS ONE
   ---------------------------------
   spk/sb-101955-118_long.bsp -- JPL small-body solution 118, long-span
   variant. 80.7 MB, body 2101955 relative to the Sun (10), covering

       1900-01-01 to 2135-10-01 (TDB)

   "This ephemeris is a final reconstruction based on ground-based radar and
   optical data since 1999, and pseudo range points from all the orbital
   phases of the OSIRIS-REx mission with at least ten days of maneuver-free
   arc from Orbital A (January 2019) to TAG (October 2020). The reference
   planetary ephemeris is DE424." -- the file's own comment area. It accounts
   for the Sun, the planets, Pluto, the Moon, 343 small-body perturbers,
   relativity, the Yarkovsky effect and other solar radiation perturbations;
   see Farnocchia et al. (2021), doi:10.1016/j.icarus.2021.114594.

       https://ssd.jpl.nasa.gov/ftp/xfr/sb-101955-118_long.bsp
       sha256 2e25094e62542f388071b43e6f7b794024ca3fa4e399306976ac62a1a5c3b48f
       84,648,960 bytes, Last-Modified 2021-11-15

   That URL is the one JPL Horizons itself prints when it declines to build a
   Bennu SPK ("SPK creation is not available for pre-computed objects in the
   major body index" -- Bennu's Horizons record is *SPECIAL*, sourced from
   ORX_merged). It is a transfer directory, not a versioned archive path, so
   scripts/fetch_kernels.py pins the sha256 above: a mismatch there may mean
   JPL replaced the file, not only that a download was corrupted.

   NAIF distributes the same solution as sb-101955-118.bsp in
   pub/naif/ORX/kernels/spk/, but only over 2015-01-01 to 2023-05-31. Measured
   against the long file over that overlap, the two agree to 4.1e-7 km
   (0.4 mm) -- same physics, shorter window -- so the long file is not a
   different product, it is the same solution carried past today.

   SUPERSEDES sb-101955-76 (2013), which is what this repository shipped as
   data/horizons_cache/bodies/101955_Bennu_(1999_RQ36)/ephemeris.spk until
   2026-08-27 and still ships for provenance; that file is byte-identical to
   NAIF's bennu_refdrmc_v1.bsp (md5 1c1b36831546de7434af83602756b962 for
   both). It covers 2015-01-01 to 2023-05-31 only, so any epoch after
   2023-05-31 -- including today -- silently left its coverage and the engine
   fell back to the config's Keplerian elements with no warning printed. At
   2026-01-01 that fallback was wrong by 0.092 AU in heliocentric distance,
   23.5 deg in solar direction, 18.3% in insolation and up to 23.9 K in
   surface temperature. Inside 2019-01-10, where both files are valid,
   solution 76 and solution 118 differ by at most 13.9 km, which moves the
   tutorial's temperatures by 2.1e-4 K.

   Shared kernels (leapseconds, planetary constants, planetary ephemeris) come
   from data/kernels/, populated by the same fetch command.

   \begindata

      PATH_VALUES     = ( 'data/kernels' )

      PATH_SYMBOLS    = ( 'KERNELS' )

      KERNELS_TO_LOAD = (

                       '$KERNELS/lsk/naif0012.tls'

                       '$KERNELS/pck/pck00011.tpc'

                       '$KERNELS/spk/de440s.bsp'

                       '$KERNELS/spk/sb-101955-118_long.bsp'

                        )

   \begintext

End of MK.
