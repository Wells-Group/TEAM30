#include "team30.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>

using namespace dolfinx;
using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char *argv[])
{
    init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    MPI_Comm comm = MPI_COMM_WORLD;

    {
        // File name
        std::string filename = "../meshes/three_phase3D_refined1.xdmf";
        std::string name = "mesh";
        bool verbose = true;

        io::XDMFFile file(comm, filename, "r");
        std::pair<mesh::CellType, int> cell = file.read_cell_type(name);
        fem::CoordinateElement<U> cmap(cell.first, cell.second);

        // Read mesh
        auto ghost_mode = dolfinx::mesh::GhostMode::none;
        auto mesh = std::make_shared<dolfinx::mesh::Mesh<double>>(file.read_mesh(cmap, ghost_mode, name));
        int tdim = mesh->topology()->dim();
        mesh->topology_mutable()->create_connectivity(tdim - 1, 0);
        mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
        mesh->topology_mutable()->create_connectivity(tdim - 2, tdim);

        // Read mesh tags
        auto ct = file.read_meshtags(*mesh, name = "Cell_markers");
        auto ft = file.read_meshtags(*mesh, name = "Facet_markers");

        // close xdmf file
        file.close();

        // Create Nedelec function space
        auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            functionspace_form_team30_a, "u", mesh));

        // Create DG0 function space
        auto DG0 = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            functionspace_form_team30_a, "mu_R", mesh));

        // Create functions
        auto u = std::make_shared<fem::Function<T>>(V);
        auto u0 = std::make_shared<fem::Function<T>>(V);

        // Create DG0 functions
        auto mu_R = std::make_shared<fem::Function<T>>(DG0);
        auto sigma = std::make_shared<fem::Function<T>>(DG0);
        auto J0z = std::make_shared<fem::Function<T>>(DG0);

        // Create constants
        auto mu_0 = std::make_shared<dolfinx::fem::Constant<T>>(4.0 * M_PI * 1e-7);
        auto dt = std::make_shared<dolfinx::fem::Constant<T>>(1e-6);

        if (verbose)
        {
            // Number of degrees of freedom
            int num_dofs = V->dofmap()->index_map->size_global() * V->dofmap()->index_map_bs();
            std::cout << "Number of dofs: " << num_dofs << std::endl;
            // Number of cells
            int num_cells = mesh->topology()->index_map(tdim)->size_global();
            std::cout << "Number of cells: " << num_cells << std::endl;
            // Number of DG0 dofs
            int num_dg0_dofs = DG0->dofmap()->index_map->size_global() * DG0->dofmap()->index_map_bs();
            std::cout << "Number of DG0 dofs: " << num_dg0_dofs << std::endl;
        }

        // Create Dirichlet boundary condition
        const std::vector<std::int32_t> facets = mesh::exterior_facet_indices(*mesh->topology());
        std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
            *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);

        auto u_D = std::make_shared<fem::Function<T>>(V);
        auto bc = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs);

        // Create LHS form
        auto cell_domains = fem::compute_integration_domains(
            fem::IntegralType::cell, *V->mesh()->topology(), ct.indices(),
            mesh->topology()->dim(), ct.values());

        const std::vector<std::shared_ptr<const fem::FunctionSpace<U>>> spaces = {V, V};
        auto a = std::make_shared<dolfinx::fem::Form<T>>(
            dolfinx::fem::create_form<T>(
                *form_team30_a, {V, V},
                {{"u", u}, {"sigma", sigma}, {"mu_R", mu_R}},
                {
                    {"mu_0", mu_0},
                    {"dt", dt},
                },
                {}));

        // Create RHS form
        auto L = std::make_shared<dolfinx::fem::Form<T>>(
            dolfinx::fem::create_form<T>(
                *form_team30_L, {V}, {{"J0z", J0z}, {"u0", u0}, {"sigma", sigma}},
                {
                    {"mu_0", mu_0},
                    {"dt", dt},
                },
                {}));

        [[maybe_unused]] Mat A = fem::petsc::create_matrix<T>(*a);
    }

    PetscFinalize();
    return 0;
}