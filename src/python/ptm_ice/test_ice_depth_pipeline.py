"""Tests for ice_depth_pipeline.py

Verifies:
- Binary streaming reader (correct mean / max aggregation)
- Legacy VTK ASCII reader (correct parsing and aggregation)
- End-to-end pipeline with minimal synthetic data

"""
import os
import tempfile
import shutil
import numpy as np
import pytest

from ptm_ice.ice_depth_pipeline import (
    stream_binary_mean_max,
    stream_vtk_mean_max,
    run_pipeline,
    write_csv,
)


# =============================== Fixtures & helpers ===============================

def write_binary_temperature_file(path: str, data: np.ndarray, dtype: str) -> None:
    data.astype(dtype).tofile(path)


def write_vtk_temperature_file(path: str, temps: np.ndarray) -> None:
    n_faces = temps.size
    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Temperature test\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"CELL_DATA {n_faces}\n")
        f.write("SCALARS temperature_0 float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in temps:
            f.write(f"{val:.6f}\n")


@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


# =============================== Tests ===============================

def test_binary_stream_mean_max(temp_dir):
    # Create synthetic binary files: 3 timesteps, 5 faces
    # Step 0: [100, 101, 102, 103, 104]
    # Step 1: [105, 106, 107, 108, 109]
    # Step 2: [110, 111, 112, 113, 114]
    # Mean should be [105, 106, 107, 108, 109]
    # Max should be [110, 111, 112, 113, 114]
    data_0 = np.array([100., 101., 102., 103., 104.], dtype='float32')
    data_1 = np.array([105., 106., 107., 108., 109.], dtype='float32')
    data_2 = np.array([110., 111., 112., 113., 114.], dtype='float32')

    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0000.bin'), data_0, 'float32')
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0001.bin'), data_1, 'float32')
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0002.bin'), data_2, 'float32')

    T_mean, T_max = stream_binary_mean_max(temp_dir, timesteps=3, dtype='float32')

    # Verify shapes
    assert T_mean.shape == (5,)
    assert T_max.shape == (5,)

    # Verify mean
    expected_mean = (data_0 + data_1 + data_2) / 3.0
    np.testing.assert_allclose(T_mean, expected_mean, atol=1e-6)

    # Verify max
    expected_max = np.maximum(data_0, np.maximum(data_1, data_2))
    np.testing.assert_allclose(T_max, expected_max, atol=1e-6)


def test_binary_stream_mean_max_parallel(temp_dir):
    """Parallel accumulation (threads>1) should match single-thread results."""
    data_0 = np.linspace(0, 4, 5).astype('float32') + 100
    data_1 = data_0 + 5
    data_2 = data_1 + 5
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0000.bin'), data_0, 'float32')
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0001.bin'), data_1, 'float32')
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0002.bin'), data_2, 'float32')
    T_mean_st, T_max_st = stream_binary_mean_max(temp_dir, timesteps=3, dtype='float32', threads=1, progress=False)
    T_mean_mt, T_max_mt = stream_binary_mean_max(temp_dir, timesteps=3, dtype='float32', threads=2, progress=False)
    np.testing.assert_allclose(T_mean_mt, T_mean_st, atol=1e-9)
    np.testing.assert_allclose(T_max_mt, T_max_st, atol=1e-9)


def test_vtk_stream_mean_max(temp_dir):
    # Single VTK file with 2 timesteps (multiple SCALARS blocks)
    n_faces = 4
    T_step0 = np.array([50., 60., 70., 80.])
    T_step1 = np.array([55., 65., 75., 85.])

    vtk_path = os.path.join(temp_dir, 'temperature_combined.vtk')
    # Write two SCALARS blocks in one file
    with open(vtk_path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Multi-step temperature\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"CELL_DATA {n_faces}\n")
        # Block 0
        f.write("SCALARS temperature_0 float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in T_step0:
            f.write(f"{val}\n")
        # Block 1
        f.write("SCALARS temperature_1 float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in T_step1:
            f.write(f"{val}\n")

    T_mean, T_max = stream_vtk_mean_max(temp_dir, timesteps=0)  # read all

    expected_mean = (T_step0 + T_step1) / 2.0
    expected_max = np.maximum(T_step0, T_step1)

    np.testing.assert_allclose(T_mean, expected_mean, atol=1e-5)
    np.testing.assert_allclose(T_max, expected_max, atol=1e-5)


def test_end_to_end_pipeline(temp_dir):
    # Create binary dataset: 2 timesteps, 3 faces
    data_0 = np.array([90., 95., 100.], dtype='float32')
    data_1 = np.array([91., 96., 101.], dtype='float32')
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0000.bin'), data_0, 'float32')
    write_binary_temperature_file(os.path.join(temp_dir, 'temperature_0001.bin'), data_1, 'float32')

    out_dir = os.path.join(temp_dir, 'output')
    csv_path = run_pipeline(
        source_dir=temp_dir,
        fmt='binary',
        timesteps=2,
        out_dir=out_dir,
        dtype='float32',
        backend='numpy',  # force NumPy to avoid Numba import issues in CI
        chunk_size=10,
    )
    assert os.path.isfile(csv_path)

    # Read CSV and check that depths exist
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 4  # header + 3 faces
    header = lines[0].strip()
    assert 'ice_depth' in header

    # Parse first data row
    row0 = lines[1].strip().split(',')
    assert row0[0] == '0'  # face index
    # Additional checks: just ensure depths are finite non-negative floats
    ice_depth = float(row0[3])
    assert ice_depth >= 0.0 or np.isnan(ice_depth)


def test_csv_writer(temp_dir):
    T_mean = np.array([100., 105., 110.])
    T_max = np.array([105., 110., 115.])
    depths = np.array([0.0012, 0.0045, np.nan])

    csv_path = os.path.join(temp_dir, 'test.csv')
    write_csv(csv_path, T_mean, T_max, depths)

    # Quick validation: read lines
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 4  # header + 3 rows
    # Check NaN representation
    last_row = lines[-1].strip()
    assert 'nan' in last_row.lower()


def test_pvd_vtu_output_with_statistics(temp_dir):
    """Verify that PVD/VTU export is generated with injected CellData arrays using statistics files."""
    # Synthetic statistics arrays (3 faces)
    T_mean = np.array([100., 110., 120.])
    T_max = np.array([130., 140., 150.])
    n_faces = T_mean.size

    # Minimal mesh template VTU (geometry + dummy CellData)
    template_path = os.path.join(temp_dir, 'mesh_template.vtu')
    with open(template_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('<UnstructuredGrid>\n')
        f.write(f'  <Piece NumberOfPoints="{n_faces}" NumberOfCells="{n_faces}">\n')
        f.write('    <PointData>\n')
        f.write('    </PointData>\n')
        f.write('    <CellData>\n')
        f.write('      <DataArray type="Float64" Name="dummy" format="ascii">\n')
        f.write('        ' + ' '.join(['0']*n_faces) + '\n')
        f.write('      </DataArray>\n')
        f.write('    </CellData>\n')
        f.write('    <Points>\n')
        f.write('      <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for i in range(n_faces):
            f.write(f'        {i} 0 0\n')
        f.write('      </DataArray>\n')
        f.write('    </Points>\n')
        f.write('    <Cells>\n')
        f.write('      <DataArray type="Int32" Name="connectivity" format="ascii">')
        f.write(' '.join(str(i) for i in range(n_faces)))
        f.write('</DataArray>\n')
        f.write('      <DataArray type="Int32" Name="offsets" format="ascii">')
        f.write(' '.join(str(i+1) for i in range(n_faces)))
        f.write('</DataArray>\n')
        f.write('      <DataArray type="UInt8" Name="types" format="ascii">' + ' '.join(['1']*n_faces) + '</DataArray>\n')
        f.write('    </Cells>\n')
        f.write('  </Piece>\n')
        f.write('</UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

    # Statistics files (avg/max) – single DataArray each
    for fname, arr in [('statistics_avg.vtu', T_mean), ('statistics_max.vtu', T_max)]:
        path = os.path.join(temp_dir, fname)
        with open(path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('<UnstructuredGrid>\n')
            f.write(f'  <Piece NumberOfPoints="{n_faces}" NumberOfCells="{n_faces}">\n')
            f.write('    <CellData>\n')
            f.write(f'      <DataArray type="Float64" Name="temp" format="ascii">\n')
            f.write('        ' + ' '.join(f'{v:.6f}' for v in arr) + '\n')
            f.write('      </DataArray>\n')
            f.write('    </CellData>\n')
            f.write('  </Piece>\n')
            f.write('</UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

    out_dir = os.path.join(temp_dir, 'viz_output')
    csv_path = run_pipeline(
        source_dir=temp_dir,
        fmt='binary',  # format irrelevant; statistics will short-circuit
        timesteps=0,
        out_dir=out_dir,
        dtype='float32',
        backend='numpy',
        pvd=True,
        mesh_vtu=template_path,
    )
    assert os.path.isfile(csv_path)
    vtu_path = os.path.join(out_dir, 'ice_depth.vtu')
    pvd_path = os.path.join(out_dir, 'ice_depth.pvd')
    assert os.path.isfile(vtu_path)
    assert os.path.isfile(pvd_path)

    with open(vtu_path, 'r') as f:
        content = f.read()
    assert 'Name="ice_depth"' in content
    assert 'Name="ice_T_mean"' in content
    assert 'Name="ice_T_max"' in content
    # PVD references VTU file
    with open(pvd_path, 'r') as f:
        pvd_content = f.read()
    assert 'ice_depth.vtu' in pvd_content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
