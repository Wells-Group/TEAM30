import dolfinx.io
from mpi4py import MPI

class PostProcessing(dolfinx.io.XDMFFile):
    """
    Post processing class adding a sligth overhead to the XDMFFile class
    """

    def __init__(self, comm: MPI.Intracomm, filename: str):
        super(PostProcessing, self).__init__(comm, f"{filename}.xdmf", "w")

    def write_function(self, u, t, name: str = None):
        if name is not None:
            u.name = name
        super(PostProcessing, self).write_function(u, t)
