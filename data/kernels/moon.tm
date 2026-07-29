KPL/MK

   PTM3D metakernel for Moon illumination calculations.

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

                       '$KERNELS/pck/moon_pa_de440_200625.bpc'

                       '$KERNELS/fk/moon_de440_250416.tf'

                        )

   \begintext

End of MK.
