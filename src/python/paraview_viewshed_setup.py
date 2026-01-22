"""
ParaView Viewshed Animation Setup for ParaView 5.13.2
"""

from paraview.simple import *
import sys

# Force output to be visible
sys.stdout.flush()
sys.stderr.flush()

print("\n" + "="*70)
print("PARAVIEW VIEWSHED SETUP STARTING")
print("="*70)

try:
    active_source = GetActiveSource()
    print(f"Active source type: {type(active_source)}")
    print(f"Active source: {active_source}")
    
    if active_source is None:
        print("ERROR: No data loaded. Please open a PVD file first.")
        print("File > Open Data > traverse_viewsheds.pvd")
    else:
        print(f"Source class: {active_source.GetClassName()}")
        
        # Try to get cell data arrays
        print("\nChecking for arrays...")
        try:
            # Get output data
            output = active_source.GetClientSideObject().GetOutputDataObject(0)
            if output:
                cell_data = output.GetCellData()
                if cell_data:
                    n_arrays = cell_data.GetNumberOfArrays()
                    print(f"Found {n_arrays} cell arrays:")
                    for i in range(n_arrays):
                        arr_name = cell_data.GetArrayName(i)
                        arr = cell_data.GetArray(i)
                        n_tuples = arr.GetNumberOfTuples()
                        print(f"  - {arr_name} ({n_tuples} values)")
                else:
                    print("No cell data found")
            else:
                print("Could not get output data object")
        except Exception as e:
            print(f"Error inspecting arrays: {e}")
            import traceback
            traceback.print_exc()
        
        # Try basic coloring
        print("\nApplying visibility coloring...")
        view = GetActiveViewOrCreate('RenderView')
        display = Show(active_source, view)
        ColorBy(display, ('CELLS', 'visibility'))
        print("✓ Colored by visibility")
        
        # Try to create threshold
        print("\nAttempting to create viewpoint threshold...")
        try:
            threshold = Threshold(
                Input=active_source,
                Scalars=['CELLS', 'viewpoint_face'],
                ThresholdMethod='Between',
                LowerThreshold=0.5,
                UpperThreshold=1.5
            )
            thresh_display = Show(threshold, view)
            # Remove colormap - use solid red color instead
            thresh_display.SetScalarBarVisibility(view, False)
            ColorBy(thresh_display, None)  # Remove colormap
            thresh_display.DiffuseColor = [1.0, 0.0, 0.0]  # Solid red
            thresh_display.AmbientColor = [0.8, 0.0, 0.0]
            thresh_display.SpecularColor = [1.0, 1.0, 1.0]
            thresh_display.EdgeVisibility = True
            thresh_display.EdgeColor = [1.0, 1.0, 0.0]  # Yellow edges
            thresh_display.LineWidth = 2.0
            thresh_display.Representation = 'Surface'
            thresh_display.Opacity = 1.0
            print("✓ Viewpoint threshold created (solid RED with yellow edges)")
        except Exception as e:
            print(f"Could not create threshold: {e}")
            print("This likely means viewpoint_face is not in the data")
        
        Render()
        print("\n✓ Rendered")

except Exception as e:
    print(f"EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("SETUP COMPLETE")
print("="*70)
sys.stdout.flush()


