KPL/MK

   PTM3D meta-kernel: asteroid (101955) Bennu, generic-body SPICE path.
   Used by docs/tutorials/notebooks/04_asteroid_spice.ipynb and
   examples/quickstart_asteroid_bennu.yaml.

   Paths are RELATIVE to the repository root -- run ptm_thermal_sim from the
   repo root (or edit PATH_VALUES to an absolute path).

   This file lives in data/kernels/ (a bare filename, like moon.tm/europa.tm/
   mercury.tm) because generic_body.spice_metakernel is resolved as
   "<compiled-in SPICE_KERNELS_DIR>/<spice_metakernel value>" (plain string
   concatenation, illumination_model.cpp initialize_spice()) -- an absolute or
   data/horizons_cache-relative value there does not resolve. The body-specific
   ephemeris.spk it loads below is NOT here; it stays with the body under
   data/horizons_cache/bodies/, resolved via the BODY path symbol instead.

   Shared kernels (leapseconds, planetary constants, planetary ephemeris) come
   from data/kernels/, populated by:  python3 scripts/fetch_kernels.py --bodies moon
   (that target also fetches two Moon-only files this metakernel does not load).

   Bennu's own ephemeris, ephemeris.spk, is not generated on demand -- JPL
   Horizons' small-body SPK service declines ad hoc
   SPK generation for Bennu ("SPK creation is not available for pre-computed
   objects in the major body index"). This is the OSIRIS-REx navigation team's
   own archived reference-trajectory SPK instead: bennu_refdrmc_v1.bsp, NAIF
   node PDS4 archive, urn:nasa:pds:orex.spice:spice_kernels:spk_bennu_refdrmc_v1,
   https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/bennu_refdrmc_v1.bsp
   Coverage: 2014-12-31 to 2023-05-30. Body id stored in the file: 2101955
   (read back with data_manager.read_spk_body_id -- do not assume it from the
   asteroid number; see the notebook's environment-check section).

   \begindata

      PATH_VALUES     = ( 'data/kernels',
                           'data/horizons_cache/bodies/101955_Bennu_(1999_RQ36)' )

      PATH_SYMBOLS    = ( 'KERNELS',
                           'BODY' )

      KERNELS_TO_LOAD = (

                       '$KERNELS/lsk/naif0012.tls'

                       '$KERNELS/pck/pck00011.tpc'

                       '$KERNELS/spk/de440s.bsp'

                       '$BODY/ephemeris.spk'

                        )

   \begintext

End of MK.
