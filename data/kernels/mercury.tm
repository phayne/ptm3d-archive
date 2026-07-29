KPL/MK

   PTM3D metakernel for Mercury illumination calculations.

   Paths are RELATIVE to the repository root — run ptm_thermal_sim from the
   repo root, or change PATH_VALUES to the absolute path of data/kernels.
   Populate this directory first with:  python3 scripts/fetch_kernels.py

   \begindata

      PATH_VALUES     = ( 'data/kernels' )

      PATH_SYMBOLS    = ( 'KERNELS' )

      KERNELS_TO_LOAD = (

                       '$KERNELS/lsk/naif0012.tls'

                       '$KERNELS/pck/pck00011.tpc'

                       '$KERNELS/spk/de440s.bsp'

                        )

   \begintext

End of MK.
