#include "team30.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/petsc.h>

using namespace dolfinx;
using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

/// @brief Create discrete gradient matrix
la::petsc::Matrix discrete_gradient(const fem::FunctionSpace<U> &V0,
                                    const fem::FunctionSpace<U> &V1)
{
    assert(V0.mesh());
    auto mesh = V0.mesh();
    assert(V1.mesh());
    assert(mesh == V1.mesh());
    MPI_Comm comm = mesh->comm();

    auto dofmap0 = V0.dofmap();
    assert(dofmap0);
    auto dofmap1 = V1.dofmap();
    assert(dofmap1);

    // Create and build  sparsity pattern
    assert(dofmap0->index_map);
    assert(dofmap1->index_map);
    la::SparsityPattern sp(
        comm, {dofmap1->index_map, dofmap0->index_map},
        {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

    int tdim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(tdim);
    assert(map);
    std::vector<std::int32_t> c(map->size_local(), 0);
    std::iota(c.begin(), c.end(), 0);
    fem::sparsitybuild::cells(sp, c, {*dofmap1, *dofmap0});
    sp.finalize();

    // Build operator
    auto A = la::petsc::Matrix(la::petsc::create_matrix(comm, sp), false);
    MatSetOption(A.mat(), MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    fem::discrete_gradient<T, U>(
        *V0.mesh()->topology_mutable(), {*V0.element(), *V0.dofmap()},
        {*V1.element(), *V1.dofmap()},
        la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES));
    // A.mat()
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
    return A;
}

/// @brief Create interpolation matrix
/// @param V0 Function space to interpolate from
/// @param V1 Function space to interpolate to
/// @return Interpolation matrix
la::petsc::Matrix interpolation_matrix(const fem::FunctionSpace<U> &V0,
                                       const fem::FunctionSpace<U> &V1)
{
    assert(V0.mesh());
    auto mesh = V0.mesh();
    assert(V1.mesh());
    assert(mesh == V1.mesh());
    MPI_Comm comm = mesh->comm();

    auto dofmap0 = V0.dofmap();
    assert(dofmap0);
    auto dofmap1 = V1.dofmap();
    assert(dofmap1);

    // Create and build  sparsity pattern
    assert(dofmap0->index_map);
    assert(dofmap1->index_map);
    dolfinx::la::SparsityPattern sp(
        comm, {dofmap1->index_map, dofmap0->index_map},
        {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

    int tdim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(tdim);
    assert(map);
    std::vector<std::int32_t> c(map->size_local(), 0);
    std::iota(c.begin(), c.end(), 0);
    dolfinx::fem::sparsitybuild::cells(sp, c, {*dofmap1, *dofmap0});
    sp.finalize();

    // Build operator
    auto A = la::petsc::Matrix(la::petsc::create_matrix(comm, sp), false);
    MatSetOption(A.mat(), MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    dolfinx::fem::interpolation_matrix<T, U>(
        V0, V1, dolfinx::la::petsc::Matrix::set_block_fn(A.mat(), INSERT_VALUES));

    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
    return A;
}

/// Given a DG-0 scalar field J_0, update it to be alpha*J*cos(omega*t + beta)
/// in the domains with copper windings
void update_current_density(std::shared_ptr<fem::Function<T>> J0z,
                            T omega_J, T t, dolfinx::mesh::MeshTags<int32_t> ct,
                            std::map<int, std::vector<T>> currents)
{
    T J0 = 1e6 * std::sqrt(2.0); // [A/m^2] Current density of copper winding
    J0z->x()->set(0.0);
    std::span array = J0z->x()->mutable_array();
    for (auto [domain, current] : currents)
    {
        T current_value = J0 * current[0] * std::cos(omega_J * t + current[1]);
        std::vector<int> cells = ct.find(domain);
        for (int cell : cells)
            array[cell] = current_value;
    }
}

int main(int argc, char *argv[])
{
    init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    MPI_Comm comm = MPI_COMM_WORLD;

    // --------------------------------------------------
    // Problem parameters
    // --------------------------------------------------
    T pi = std::acos(-1);
    int num_phases = 3;
    int steps_per_phase = 100;
    T freq = 60; // Hz - Frequency of excitation
    [[maybe_unused]] T T_ = num_phases * 1 / freq;
    [[maybe_unused]] T dt_ = 1.0 / steps_per_phase * 1 / freq;
    T omega_J = 2 * pi * freq; // [rad/s] Angular frequency of excitation

    // Map domains to cell markers
    std::map<std::string, std::vector<int>> domains = {
        {"Cu", {7, 8, 9, 10, 11, 12}},
        {"Stator", {6}},
        {"Rotor", {5}},
        {"Al", {4}},
        {"AirGap", {2, 3}},
        {"Air", {1}}};

    // Map cell markers to currents cos(omega*t + beta)
    std::map<int, std::vector<T>> currents = {
        {7, {1.0, 0.0}},
        {8, {-1.0, 2 * pi / 3}},
        {9, {1.0, 4 * pi / 3}},
        {10, {-1.0, 0.0}},
        {11, {1.0, 2 * pi / 3}},
        {12, {-1.0, 4 * pi / 3}}};

    T sigma_non_conducting = 1e-5;
    std::map<std::string, T> mu_r_def = {
        {"Cu", 1}, {"Stator", 30}, {"Rotor", 30}, {"Al", 1}, {"Air", 1}, {"AirGap", 1}};
    std::map<std::string, T> sigma_def = {
        {"Cu", sigma_non_conducting},
        {"Stator", sigma_non_conducting},
        {"Rotor", 1.6e6},
        {"Al", 3.72e7},
        {"Air", sigma_non_conducting},
        {"AirGap", sigma_non_conducting}};

    // --------------------------------------------------//

    {
        // File name
        std::string filename = "../meshes/three_phase3D.xdmf";
        std::string name = "mesh";
        bool verbose = true;

        // Open xdmf file and read cell type
        io::XDMFFile file(comm, filename, "r");
        std::pair<mesh::CellType, int> cell = file.read_cell_type(name);
        fem::CoordinateElement<U> cmap(cell.first, cell.second);

        // Read mesh from file
        auto ghost_mode = dolfinx::mesh::GhostMode::none;
        auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(file.read_mesh(cmap, ghost_mode, name));
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
        auto V_A = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            functionspace_form_team30_a00, "u_A", mesh));

        // Create DG0 function space
        auto DG0 = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            functionspace_form_team30_a00, "mu_R", mesh));

        auto V_V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            functionspace_form_team30_a11, "u_V", mesh));

        // // Create functions
        auto u_A = std::make_shared<fem::Function<T>>(V_A);
        auto u_A0 = std::make_shared<fem::Function<T>>(V_A);
        auto u_V = std::make_shared<fem::Function<T>>(V_V);

        // Create DG0 functions
        auto mu_R = std::make_shared<fem::Function<T>>(DG0);
        auto sigma = std::make_shared<fem::Function<T>>(DG0);
        auto J0z = std::make_shared<fem::Function<T>>(DG0);

        // Create constants
        auto mu_0 = std::make_shared<fem::Constant<T>>(4.0 * M_PI * 1e-7);
        auto dt = std::make_shared<fem::Constant<T>>(0.00016666666666666666);
        auto zero = std::make_shared<fem::Constant<T>>(0.0);

        // Set material properties mu_R and sigma
        for (auto [material, markers] : domains)
        {
            for (auto marker : markers)
            {
                std::vector<int> cells = ct.find(marker);
                for (int cell : cells)
                {
                    mu_R->x()->mutable_array()[cell] = mu_r_def[material];
                    sigma->x()->mutable_array()[cell] = sigma_def[material];
                }
            }
        }

        if (verbose)
        {
            // Number of degrees of freedom
            int num_dofs = V_A->dofmap()->index_map->size_global() * V_A->dofmap()->index_map_bs();
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

        // Boundary condition for u_A
        std::vector<std::int32_t> bdofs_A = fem::locate_dofs_topological(
            *V_A->mesh()->topology_mutable(), *V_A->dofmap(), tdim - 1, facets);

        auto u_D = std::make_shared<fem::Function<T>>(V_A);
        u_D->x()->set(0.0);
        auto bc0 = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs_A);

        // Boudary condition for u_V
        std::vector<std::int32_t> bdofs_V = fem::locate_dofs_topological(
            *V_V->mesh()->topology_mutable(), *V_V->dofmap(), tdim - 1, facets);
        auto bc1 = std::make_shared<const fem::DirichletBC<T>>(zero, bdofs_V, V_V);

        // const std::vector<std::shared_ptr<const fem::DirichletBC<T, U>>> bcs = {bc0, bc1};

        auto a00 = std::make_shared<fem::Form<T>>(
            fem::create_form<T>(
                *form_team30_a00, {V_A, V_A},
                {{"u_A", u_A}, {"sigma", sigma}, {"mu_R", mu_R}},
                {
                    {"mu_0", mu_0},
                    {"dt", dt},
                },
                {}));

        auto a01 = std::make_shared<fem::Form<T>>(
            fem::create_form<T>(
                *form_team30_a01, {V_A, V_V},
                {{"sigma", sigma}},
                {
                    {"mu_0", mu_0},
                },
                {}));

        auto a10 = std::make_shared<fem::Form<T>>(
            fem::create_form<T>(
                *form_team30_a10, {V_V, V_A},
                {}, {
                        {"zero", zero},
                    },
                {}));

        auto a11 = std::make_shared<fem::Form<T>>(
            fem::create_form<T>(
                *form_team30_a11, {V_V, V_V},
                {{"sigma", sigma}},
                {
                    {"mu_0", mu_0},
                },
                {}));

        std::vector<const fem::Form<PetscScalar, T> *> a0row = {&(*a00), &(*a01)};
        std::vector<const fem::Form<PetscScalar, T> *> a1row = {&(*a10), &(*a11)};
        const std::vector<std::vector<const fem::Form<PetscScalar, T> *>> &a = {a0row, a1row};
        auto A = la::petsc::Matrix(fem::petsc::create_matrix_nest(a, {}), false);

        for (PetscInt idxm = 0; idxm < 2; idxm++)
        {
            for (PetscInt jdxm = 0; jdxm < 2; jdxm++)
            {

                Mat sub;
                MatNestGetSubMat(A.mat(), idxm, jdxm, &sub);
                MatSetOption(sub, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
                fem::assemble_matrix(la::petsc::Matrix::set_fn(sub, ADD_VALUES), *a[idxm][jdxm], {bc0, bc1});
                MatAssemblyBegin(sub, MAT_FLUSH_ASSEMBLY);
                MatAssemblyEnd(sub, MAT_FLUSH_ASSEMBLY);
                if (idxm == jdxm)
                {
                    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(sub, INSERT_VALUES), *a[idxm][jdxm]->function_spaces()[0], {bc0, bc1});
                }
            }
        }

        Mat mat = A.mat();
        MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
        MatView(mat, PETSC_VIEWER_STDOUT_WORLD);

        Mat sub;
        MatNestGetSubMat(A.mat(), 0, 0, &sub);
        PetscReal nrm = 0;
        MatNorm(sub, NORM_FROBENIUS, &nrm);
        std::cout << "\nNorm of matrix: " << nrm << std::endl;

        // // Create RHS form
        // auto L = std::make_shared<fem::Form<T>>(
        //     fem::create_form<T>(
        //         *form_team30_L, {V}, {{"J0z", J0z}, {"u0", u0}, {"sigma", sigma}},
        //         {
        //             {"mu_0", mu_0},
        //             {"dt", dt},
        //         },
        //         {}));

        // std::cout << "Assembling matrix and vector" << std::endl;

        // // Create matrix and RHS vector
        // [[maybe_unused]] auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
        // la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
        //                 L->function_spaces()[0]->dofmap()->index_map_bs());

        // // Assemble matrix
        // MatZeroEntries(A.mat());
        // fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
        //                      *a, {bc});
        // MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
        // MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
        // fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
        //                      {bc});
        // MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
        // MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

        // // Assemble vector
        // b.set(0.0);
        // fem::assemble_vector(b.mutable_array(), *L);
        // fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1.0));
        // b.scatter_rev(std::plus<T>());
        // fem::set_bc<T, U>(b.mutable_array(), {bc});

        // std::cout << "Creating PETSc solver" << std::endl;
        // // Create linear solver
        // la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
        // la::petsc::options::set("ksp_type", "gmres");
        // la::petsc::options::set("ksp_atol", 1e-8);
        // la::petsc::options::set("ksp_rtol", 1e-8);
        // la::petsc::options::set("ksp_norm_type", "unpreconditioned");
        // la::petsc::options::set("ksp_max_it", 100);
        // la::petsc::options::set("ksp_gmres_restart", 100);
        // la::petsc::options::set("ksp_diagonal_scale", true);
        // la::petsc::options::set("pc_type", "hypre");
        // la::petsc::options::set("pc_hypre_type", "ams");
        // la::petsc::options::set("pc_hypre_ams_cycle_type", 1);
        // la::petsc::options::set("pc_hypre_ams_tol", 1e-8);
        // solver.set_from_options();
        // solver.set_operator(A.mat());

        // // Create scalar function space Lagrange function space
        // // Create a Basix continuous Lagrange element of degree 1
        // basix::FiniteElement e = basix::create_element<U>(
        //     basix::element::family::P,
        //     mesh::cell_type_to_basix_type(mesh::CellType::tetrahedron), 1,
        //     basix::element::lagrange_variant::unset,
        //     basix::element::dpc_variant::unset, false);

        // // Create a scalar function space
        // auto S = std::make_shared<fem::FunctionSpace<U>>(
        //     fem::create_functionspace(mesh, e));

        // // Create Vector function space
        // auto Q = std::make_shared<fem::FunctionSpace<U>>(
        //     fem::create_functionspace(mesh, e, std::vector<std::size_t>{3}));

        // // Create discrete gradient matrix
        // auto G = discrete_gradient(*S, *V);

        // // Create interpolation matrix
        // auto I = interpolation_matrix(*Q, *V);

        // // [[maybe_unused]] KSP ksp = solver.ksp();
        // PC prec;
        // KSPGetPC(solver.ksp(), &prec);

        // PCHYPRESetDiscreteGradient(prec, G.mat());
        // // For AMS, only Nedelec interpolation matrices are needed, the
        // // Raviart-Thomas interpolation matrices can be set to NULL.
        // PCHYPRESetInterpolations(prec, tdim, NULL, NULL, I.mat(), NULL);

        // KSPSetUp(solver.ksp());
        // PCSetUp(prec);

        // // Create petsc wrapper for u and b
        // la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
        // la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);

        // // Create expression for B field
        // auto bfield_expr = fem::create_expression<T, U>(
        //     *expression_team30_B_3D, {{"u0", u}}, {});

        // // Create B field in Vector function space
        // auto B = std::make_shared<fem::Function<T>>(Q);
        // B->interpolate(bfield_expr);

        // // Open xdmf file to write results
        // io::XDMFFile out(comm, "results.xdmf", "w");
        // out.write_mesh(*mesh);

        // int rank = dolfinx::MPI::rank(comm);
        // if (rank == 0)
        //     std::cout << "Starting time loop" << std::endl;

        // auto w = std::make_shared<fem::Function<T>>(Q);

        // T t = 0.0;
        // for (int i = 0; i < 1; i++)
        // {
        //     update_current_density(J0z, omega_J, t, ct, currents);
        //     w->sub(2).interpolate(*J0z);
        //     out.write_function(*w, t);

        //     // Update RHS
        //     b.set(0.0);
        //     fem::assemble_vector(b.mutable_array(), *L);
        //     fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1.0));
        //     b.scatter_rev(std::plus<T>());
        //     fem::set_bc<T, U>(b.mutable_array(), {bc});

        //     // Solve linear system
        //     solver.solve(_u.vec(), _b.vec());
        //     u->x()->scatter_fwd();

        //     KSPConvergedReason reason;
        //     KSPGetConvergedReason(solver.ksp(), &reason);

        //     std::cout << "Converged reason: " << reason << "\n";
        //     PetscInt its;
        //     KSPGetIterationNumber(solver.ksp(), &its);

        //     std::cout << "Number of iterations: " << its << "\n";

        //     // Update u0
        //     std::copy(u->x()->array().begin(), u->x()->array().end(), u0->x()->mutable_array().begin());

        //     // Update bfield
        //     B->interpolate(bfield_expr);

        //     // Update time
        //     t += dt->value[0];
        // }

        // out.close();
    }

    PetscFinalize();
    return 0;
}