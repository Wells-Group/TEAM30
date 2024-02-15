#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/petsc.h>

#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>

#include "utils.hpp"
#include "team30.h"

using namespace dolfinx;
using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

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

    T sigma_non_conducting = 1e-5; // [S/m] Non-conducting material
    std::map<std::string, T> mu_r_def = {
        {"Cu", 1}, {"Stator", 30}, {"Rotor", 30}, {"Al", 1}, {"Air", 1}, {"AirGap", 1}};
    std::map<std::string, T> sigma_def = {
        {"Cu", sigma_non_conducting},
        {"Stator", sigma_non_conducting},
        {"Rotor", 1.6e6},
        {"Al", 3.72e7},
        {"Air", sigma_non_conducting},
        {"AirGap", sigma_non_conducting}};

    std::vector<std::string> non_conducting_materials = {"Cu", "Stator", "Air", "AirGap"};
    std::vector<std::string> conducting_materials = {"Rotor", "Al"};

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
        auto topology = mesh->topology_mutable();
        int tdim = topology->dim();
        topology->create_connectivity(tdim - 1, 0);
        topology->create_connectivity(tdim - 1, tdim);
        topology->create_connectivity(tdim - 2, tdim);
        topology->create_connectivity(tdim, tdim);

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

        // Create scalar function space (Lagrange P1)
        auto V_V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
            functionspace_form_team30_a11, "u_V", mesh));

        // Create functions for solution and previous solution
        auto u_A = std::make_shared<fem::Function<T>>(V_A);
        auto u_A0 = std::make_shared<fem::Function<T>>(V_A);
        auto u_V = std::make_shared<fem::Function<T>>(V_V);

        // Create DG0 functions for material properties
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

        if (verbose and dolfinx::MPI::rank(comm) == 0)
        {
            // Number of degrees of freedom
            int num_dofs = V_A->dofmap()->index_map->size_global() * V_A->dofmap()->index_map_bs();
            std::cout << "Number of dofs (A): " << num_dofs << std::endl;

            num_dofs = V_V->dofmap()->index_map->size_global() * V_V->dofmap()->index_map_bs();
            std::cout << "Number of dofs (V): " << num_dofs << std::endl;

            // Number of cells
            int num_cells = mesh->topology()->index_map(tdim)->size_global();
            std::cout << "Number of cells: " << num_cells << std::endl;
            // Number of DG0 dofs
            int num_dg0_dofs = DG0->dofmap()->index_map->size_global() * DG0->dofmap()->index_map_bs();
            std::cout << "Number of DG0 dofs: " << num_dg0_dofs << std::endl;
        }

        // --------------------------------------------------
        // Create Dirichlet boundary condition
        const std::vector<std::int32_t> facets = mesh::exterior_facet_indices(*mesh->topology());

        // Boundary condition for u_A
        int fdim = tdim - 1;
        std::vector<std::int32_t> bdofs_A = fem::locate_dofs_topological(
            *topology, *V_A->dofmap(), fdim, facets);

        auto u_D = std::make_shared<fem::Function<T>>(V_A);
        u_D->x()->set(0.0);
        auto bc0 = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs_A);

        // Locate all dofs in non conducting materials
        std::vector<std::int32_t> dofs_omega_n;
        for (auto material : non_conducting_materials)
        {
            std::vector<int> markers = domains[material];
            for (int marker : markers)
            {
                std::vector<int> cells = ct.find(marker);
                auto dofs = fem::locate_dofs_topological(*topology, *V_V->dofmap(), tdim, cells);
                dofs_omega_n.insert(dofs_omega_n.end(), dofs.begin(), dofs.end());
            }
        }
        // Sort and remove duplicates
        std::sort(dofs_omega_n.begin(), dofs_omega_n.end());
        dofs_omega_n.erase(std::unique(dofs_omega_n.begin(), dofs_omega_n.end()), dofs_omega_n.end());

        PetscPrintf(comm, "Number of dofs in non conducting materials: %d\n", int(dofs_omega_n.size()));

        // Locate all dofs in conducting materials
        std::vector<std::int32_t> dofs_omega_c;
        for (auto material : conducting_materials)
        {
            std::vector<int> markers = domains[material];
            for (int marker : markers)
            {
                std::vector<int> cells = ct.find(marker);
                auto dofs = fem::locate_dofs_topological(*topology, *V_V->dofmap(), tdim, cells);
                dofs_omega_c.insert(dofs_omega_c.end(), dofs.begin(), dofs.end());
            }
        }
        // Sort and remove duplicates
        std::sort(dofs_omega_c.begin(), dofs_omega_c.end());
        dofs_omega_c.erase(std::unique(dofs_omega_c.begin(), dofs_omega_c.end()), dofs_omega_c.end());

        // Create Dirichlet boundary condition for u_V
        auto bc1 = std::make_shared<const fem::DirichletBC<T>>(zero, dofs_omega_n, V_V);

        // --------------------------------------------------
        // Vector of boundary conditions
        const std::vector<std::shared_ptr<const fem::DirichletBC<T, U>>> bcs = {bc0, bc1};

        // --------------------------------------------------
        // Create forms

        // a = [[a00, a01], [a10, a11]]
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

        // FIXME: Zero block, do we need to create a form for it?
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

        std::vector<std::vector<const fem::Form<PetscScalar, T> *>> a = {{&(*a00), &(*a01)}, {&(*a10), &(*a11)}};
        auto A = la::petsc::Matrix(fem::petsc::create_matrix_nest(a, {}), false);

        std::array<PetscInt, 2> shape;
        shape[0] = a.size();
        shape[1] = a[0].size();

        std::cout << "Shape: " << shape[0] << " " << shape[1] << std::endl;

        for (PetscInt idxm = 0; idxm < shape[0]; idxm++)
        {
            for (PetscInt jdxm = 0; jdxm < shape[1]; jdxm++)
            {
                Mat Aij;
                MatNestGetSubMat(A.mat(), idxm, jdxm, &Aij);
                MatSetOption(Aij, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
                fem::assemble_matrix(la::petsc::Matrix::set_fn(Aij, ADD_VALUES), *a[idxm][jdxm], bcs);
                MatAssemblyBegin(Aij, MAT_FLUSH_ASSEMBLY);
                MatAssemblyEnd(Aij, MAT_FLUSH_ASSEMBLY);
                if (idxm == jdxm)
                    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(Aij, INSERT_VALUES), *a[idxm][jdxm]->function_spaces()[0], bcs);
                MatAssemblyBegin(Aij, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(Aij, MAT_FINAL_ASSEMBLY);
            }
        }

        // MatSetVecType(A.mat(), VECNEST);

        MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

        MatView(A.mat(), PETSC_VIEWER_STDOUT_WORLD);

        // Set the vector type the matrix will return with createVec
        PetscPrintf(comm, "Matrix A assembled\n\n");

        // Check norm of each block
        for (PetscInt idxm = 0; idxm < shape[0]; idxm++)
        {
            for (PetscInt jdxm = 0; jdxm < shape[1]; jdxm++)
            {
                Mat sub;
                MatNestGetSubMat(A.mat(), idxm, jdxm, &sub);
                PetscReal nrm = 0;
                MatNorm(sub, NORM_FROBENIUS, &nrm);
                PetscPrintf(comm, "Norm of block (%d, %d): %f\n", idxm, jdxm, nrm);
            }
        }

        // Create RHS form
        auto L0 = std::make_shared<fem::Form<T>>(
            fem::create_form<T>(
                *form_team30_L0, {V_A}, {{"J0z", J0z}, {"u_A0", u_A0}, {"sigma", sigma}},
                {
                    {"mu_0", mu_0},
                    {"dt", dt},
                },
                {}));

        auto L1 = std::make_shared<fem::Form<T>>(
            fem::create_form<T>(
                *form_team30_L1, {V_V}, {},
                {
                    {"zero", zero},
                },
                {}));

        std::vector<const fem::Form<PetscScalar, T> *> L = {&(*L0), &(*L1)};

        std::vector<
            std::pair<std::reference_wrapper<const common::IndexMap>, int>>
            _maps;

        _maps.push_back({*V_A->dofmap()->index_map, V_A->dofmap()->index_map_bs()});
        _maps.push_back({*V_V->dofmap()->index_map, V_V->dofmap()->index_map_bs()});

        KSP solver;
        KSPCreate(comm, &solver);
        KSPSetOperators(solver, A.mat(), A.mat());
        std::string prefix = "main_";
        KSPSetOptionsPrefix(solver, prefix.c_str());

        KSPSetType(solver, KSPGMRES);
        KSPSetTolerances(solver, 1e-8, 1e-8, 1e8, 100);
        KSPSetNormType(solver, KSP_NORM_UNPRECONDITIONED);

        PC pc;
        KSPGetPC(solver, &pc);
        PCSetType(pc, PCFIELDSPLIT);
        PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE);

        // FIXME: use row or col?
        IS rows[2];
        IS cols[2];
        MatNestGetISs(A.mat(), rows, cols);
        PCFieldSplitSetIS(pc, "A", rows[0]);
        PCFieldSplitSetIS(pc, "V", rows[1]);

        KSP *subksp;
        int nsplits;
        PCFieldSplitGetSubKSP(pc, &nsplits, &subksp);

        Mat A00;
        MatNestGetSubMat(A.mat(), 0, 0, &A00);
        KSPSetType(subksp[0], KSPPREONLY);
        KSPSetOperators(subksp[0], A00, A00);
        std::string prefix0 = "ksp_A_";
        KSPSetOptionsPrefix(subksp[0], prefix0.c_str());

        PC subpc0;
        KSPGetPC(subksp[0], &subpc0);

        std::map<std::string, std::string> options_A = {
            {"-pc_type", "hypre"},
            {"-pc_hypre_type", "ams"},
            {"-pc_hypre_ams_cycle_type", "1"},
            {"-pc_hypre_ams_tol", "1e-8"},
            {"-ksp_type", "preonly"}};

        PetscOptionsPrefixPush(NULL, prefix0.c_str());
        for (auto [key, value] : options_A)
        {
            PetscOptionsSetValue(NULL, key.c_str(), value.c_str());
        }
        PetscOptionsPrefixPop(NULL);
        KSPSetFromOptions(subksp[0]);

        // Create a Basix continuous Lagrange element of degree 1
        basix::FiniteElement e = basix::create_element<U>(
            basix::element::family::P,
            mesh::cell_type_to_basix_type(mesh::CellType::tetrahedron), 1,
            basix::element::lagrange_variant::unset,
            basix::element::dpc_variant::unset, false);

        // Create a scalar function space
        auto S = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(mesh, e));
        // Create Vector function space
        std::vector<std::size_t> shpe{3};
        auto Q = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(mesh, e, shpe));

        // Create discrete gradient matrix
        auto G = discrete_gradient<T, U>(*S, *V_A);
        PCHYPRESetDiscreteGradient(subpc0, G.mat());

        // Create interpolation matrix
        auto I = interpolation_matrix<T, U>(*Q, *V_A);
        PCHYPRESetInterpolations(subpc0, tdim, NULL, NULL, I.mat(), NULL);

        PCSetUp(subpc0);
        KSPSetUp(subksp[0]);

        Vec b0, x0;
        MatCreateVecs(A00, &b0, &x0);

        Mat A11;
        MatNestGetSubMat(A.mat(), 1, 1, &A11);
        KSPSetType(subksp[1], KSPPREONLY);
        KSPSetOperators(subksp[1], A11, A11);
        std::string prefix1 = "ksp_V_";
        KSPSetOptionsPrefix(subksp[1], prefix1.c_str());

        PC subpc1;
        KSPGetPC(subksp[1], &subpc1);
        PCSetType(subpc1, "gamg");
        KSPSetFromOptions(subksp[1]);
        PCSetUp(subpc1);
        KSPSetUp(subksp[1]);

        Vec b1, x1;
        MatCreateVecs(A11, &b1, &x1);
        VecSetRandom(b1, NULL);
        VecSet(x1, 0.0);

        KSPSetFromOptions(solver);
        KSPSetUp(solver);

        KSPView(solver, PETSC_VIEWER_STDOUT_WORLD);

        Vec b, x;
        MatCreateVecs(A.mat(), &b, &x);
        VecSetRandom(b, NULL);
        VecSet(x, 0.0);

        KSPSolve(solver, b, x);

        // KSPView(solver, PETSC_VIEWER_STDOUT_WORLD);

        //     // KSP *subksp;
        //     // int nsplits;
        //     // PCFieldSplitGetSubKSP(prec, &nsplits, &subksp);

        //     // KSPSetType(subksp[0], KSPPREONLY);
        //     // Mat A00;
        //     // MatNestGetSubMat(A.mat(), 0, 0, &A00);
        //     // PC subprec0;
        //     // KSPGetPC(subksp[0], &subprec0);
        //     // PCSetOperators(subprec0, A00, A00);
        //     // PCSetType(subprec0, PCLU);
        //     // PCSetUp(subprec0);

        //     // KSPSetType(subksp[1], KSPPREONLY);
        //     // Mat A11;
        //     // MatNestGetSubMat(A.mat(), 1, 1, &A11);
        //     // PC subprec1;
        //     // KSPGetPC(subksp[1], &subprec1);
        //     // PCSetOperators(subprec1, A11, A11);
        //     // PCSetType(subprec1, PCLU);
        //     // PCSetUp(subprec1);
        //     // KSPSetUp(solver);
        //     // KSPView(solver, PETSC_VIEWER_STDOUT_WORLD);

        //     Vec x1;
        //     VecDuplicate(b1, &x1);
        //     VecSet(x1, 0.0);
        //     // VecSetRandom(b1, NULL);
        //     KSPSolve(solver, b1, x1);

        //     // auto I = interpolation_matrix(*Q, *V);

        //     // // [[maybe_unused]] KSP ksp = solver.ksp();
        //     // PC prec;
        //     // KSPGetPC(solver.ksp(), &prec);

        //     // PCHYPRESetDiscreteGradient(prec, G.mat());
        //     // // For AMS, only Nedelec interpolation matrices are needed, the
        //     // // Raviart-Thomas interpolation matrices can be set to NULL.
        //     // PCHYPRESetInterpolations(prec, tdim, NULL, NULL, I.mat(), NULL);

        //     // KSPSetUp(solver.ksp());
        //     // PCSetUp(prec);

        //     // // Create petsc wrapper for u and b
        //     // la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
        //     // la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);

        //     // // Create expression for B field
        //     // auto bfield_expr = fem::create_expression<T, U>(
        //     //     *expression_team30_B_3D, {{"u0", u}}, {});

        //     // // Create B field in Vector function space
        //     // auto B = std::make_shared<fem::Function<T>>(Q);
        //     // B->interpolate(bfield_expr);

        //     // // Open xdmf file to write results
        //     // io::XDMFFile out(comm, "results.xdmf", "w");
        //     // out.write_mesh(*mesh);

        //     // int rank = dolfinx::MPI::rank(comm);
        //     // if (rank == 0)
        //     //     std::cout << "Starting time loop" << std::endl;

        //     // auto w = std::make_shared<fem::Function<T>>(Q);

        //     // T t = 0.0;
        //     // for (int i = 0; i < 1; i++)
        //     // {
        //     //     update_current_density(J0z, omega_J, t, ct, currents);
        //     //     w->sub(2).interpolate(*J0z);
        //     //     out.write_function(*w, t);

        //     //     // Update RHS
        //     //     b.set(0.0);
        //     //     fem::assemble_vector(b.mutable_array(), *L);
        //     //     fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1.0));
        //     //     b.scatter_rev(std::plus<T>());
        //     //     fem::set_bc<T, U>(b.mutable_array(), {bc});

        //     //     // Solve linear system
        //     //     solver.solve(_u.vec(), _b.vec());
        //     //     u->x()->scatter_fwd();

        //     //     KSPConvergedReason reason;
        //     //     KSPGetConvergedReason(solver.ksp(), &reason);

        //     //     std::cout << "Converged reason: " << reason << "\n";
        //     //     PetscInt its;
        //     //     KSPGetIterationNumber(solver.ksp(), &its);

        //     //     std::cout << "Number of iterations: " << its << "\n";

        //     //     // Update u0
        //     //     std::copy(u->x()->array().begin(), u->x()->array().end(), u0->x()->mutable_array().begin());

        //     //     // Update bfield
        //     //     B->interpolate(bfield_expr);

        //     //     // Update time
        //     //     t += dt->value[0];
        //     // }

        //     // out.close();
    }

    PetscFinalize();
    return 0;
}