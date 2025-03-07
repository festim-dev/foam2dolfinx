from foam2dolfinx import OpenFOAMReader
from dolfinx.io import VTXWriter
from pyvista import examples
from mpi4py import MPI


def test_reading_and_writing_cavity_example(tmpdir):
    my_of_reader = OpenFOAMReader(filename=examples.download_cavity(load=False))
    vel = my_of_reader.create_dolfinx_function(t=2.5)

    output_filename = str(tmpdir.join("velocity_points_alt_test.bp"))
    writer = VTXWriter(MPI.COMM_WORLD, output_filename, vel, engine="BP5")
    writer.write(t=0)
