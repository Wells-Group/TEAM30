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
#include "losses.h"

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

    T sigma_non_conducting = 1e-4; // [S/m] Non-conducting material
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

        // // Scale mesh by 1e-3
        // std::span<double> points = mesh->geometry().x();
        // std::transform(points.begin(), points.end(), points.begin(), [](double x) { return x * 1e-3; });

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

        T dt_ = 1.0 / steps_per_phase * 1 / freq;
        auto dt = std::make_shared<fem::Constant<T>>(dt_);
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

        // --------------------------------------------------
        // Boundary condition for u_V
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
        // Combine boundary conditions for u_A and u_V in a vector
        const std::vector<std::shared_ptr<const fem::DirichletBC<T, U>>> bcs = {bc0, bc1};

        // --------------------------------------------------
        // Create form for losses

        std::vector<std::pair<int, std::vector<std::int32_t>>> cell_domains = fem::compute_integration_domains(fem::IntegralType::cell, *topology, ct.indices(), ct.dim(), ct.values());

        std::map<
            fem::IntegralType,
            std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
            domains;
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>> temp;
        for (auto &[id, domains_] : cell_domains)
        {
            PetscPrintf(comm, "Cell domain %d\n", id);
            temp.push_back({id, std::span(domains_.data(), domains_.size())});
        }
        domains.insert({fem::IntegralType::cell, temp});

        auto LossForm = std::make_shared<fem::Form<T>>(
            fem::create_form<T, U>(
                *form_losses_q, {}, {{"An", u_A}, {"A", u_A0}, {"sigma", sigma}},
                {
                    {"dt", dt},
                },
                domains, mesh));

        T loss = fem::assemble_scalar(*LossForm);
        PetscPrintf(comm, "Loss: %f\n", loss);

        // --------------------------------------------------
        // Create bilinear forms
        // a = [[a00, a01],
        //      [a10, a11]]
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

        // --------------------------------------------------
        // Create matrix and assemble
        std::vector<std::vector<const fem::Form<PetscScalar, T> *>> a = {{&(*a00), &(*a01)}, {&(*a10), &(*a11)}};
        auto A = la::petsc::Matrix(fem::petsc::create_matrix_nest(a, {}), false);
        // MatSetVecType(A.mat(), VECNEST);

        std::array<PetscInt, 2> shape;
        shape[0] = a.size();
        shape[1] = a[0].size();

        // Assemble each block of the matrix
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

        MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

        MatView(A.mat(), PETSC_VIEWER_STDOUT_WORLD);

        // Set the vector type the matrix will return with createVec
        PetscPrintf(comm, "Matrix A assembled\n\n");

        // Check norm of each block and print it
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

        // --------------------------------------------------
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

        // --------------------------------------------------
        // Assemble RHS
        // Create Vector Nest
        std::vector<
            std::pair<std::reference_wrapper<const common::IndexMap>, int>>
            maps;
        maps.push_back({std::ref(*V_A->dofmap()->index_map), V_A->dofmap()->index_map_bs()});
        maps.push_back({std::ref(*V_V->dofmap()->index_map), V_V->dofmap()->index_map_bs()});

        // Update current value
        update_current_density<T>(J0z, omega_J, 0, ct, currents);

        Vec b = fem::petsc::create_vector_nest(maps);
        Vec x = fem::petsc::create_vector_nest(maps);

        const std::vector<std::shared_ptr<const fem::Form<PetscScalar, double>>> a_diag = {a00, a11};
        assemble_vector_nest<T>(b, L, a_diag, bcs);

        // --------------------------------------------------
        // Create Solver
        KSP solver;
        KSPCreate(comm, &solver);
        KSPSetOperators(solver, A.mat(), A.mat());
        std::string prefix = "main_";
        KSPSetOptionsPrefix(solver, prefix.c_str());

        // Set solver type
        KSPSetType(solver, KSPGMRES);
        KSPSetTolerances(solver, 1e-8, 1e-8, 1e8, 100);
        KSPSetNormType(solver, KSP_NORM_UNPRECONDITIONED);

        // Set preconditioner type
        PC pc;
        KSPGetPC(solver, &pc);
        PCSetType(pc, PCFIELDSPLIT);
        PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE);

        // Set fieldsplit options
        IS rows[2];
        IS cols[2];
        MatNestGetISs(A.mat(), rows, cols);
        PCFieldSplitSetIS(pc, "A", rows[0]); // Set the row indices for the A block
        PCFieldSplitSetIS(pc, "V", rows[1]); // Set the row indices for the V block

        // Get subksp and subpc
        KSP *subksp;
        int nsplits;
        PCFieldSplitGetSubKSP(pc, &nsplits, &subksp);

        // --------------------------------------------------
        // Set options for subksp for "A" block
        Mat A00;
        MatNestGetSubMat(A.mat(), 0, 0, &A00);
        KSPSetType(subksp[0], KSPPREONLY);
        KSPSetOperators(subksp[0], A00, A00);
        std::string prefix0 = "Ablock_";
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

        // Create a Basix continuous Lagrange element of degree 1 Used in the
        // preconditioner, maybe we can use the same element as the one used in
        // "V" space
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

        // --------------------------------------------------
        // Set options for subksp for "V" block
        // Use GAMG for the "V" block - Poisson-like problem
        Mat A11;
        MatNestGetSubMat(A.mat(), 1, 1, &A11);
        KSPSetType(subksp[1], KSPPREONLY);
        KSPSetOperators(subksp[1], A11, A11);
        std::string prefix1 = "VBlock_";
        KSPSetOptionsPrefix(subksp[1], prefix1.c_str());

        PC subpc1;
        KSPGetPC(subksp[1], &subpc1);
        PCSetType(subpc1, "gamg");
        KSPSetFromOptions(subksp[1]);
        PCSetUp(subpc1);
        KSPSetUp(subksp[1]);

        KSPSetFromOptions(solver);
        KSPSetUp(solver);
        KSPView(solver, PETSC_VIEWER_STDOUT_WORLD);
        // --------------------------------------------------
        // Test solver
        Vec b_, x_;
        MatCreateVecs(A.mat(), &b_, &x_);
        VecSet(x_, 0.0);
        VecSet(b_, 0.0);
        copy(b, b_);
        KSPSolve(solver, b_, x_);
        copy(x_, x);

        {
            Vec xi;
            VecNestGetSubVec(x, 0, &xi);
            copy<T>(xi, *u_A->x());
        }
        // --------------------------------------------------

        // Create expression for B field
        auto bfield_expr = fem::create_expression<T, U>(
            *expression_team30_B_3D, {{"A_out", u_A}}, {});

        // Create B field in Vector function space
        auto B = std::make_shared<fem::Function<T>>(Q);
        B->interpolate(bfield_expr);
        B->name = "B";

        io::XDMFFile out(comm, "results.xdmf", "w");
        out.write_mesh(*mesh);
        // out.write_function(*B, 0.0);

        int rank = dolfinx::MPI::rank(comm);
        if (rank == 0)
            std::cout << "Starting time loop" << std::endl;

        // Create function for J0z in 3D
        auto J = std::make_shared<fem::Function<T>>(Q);
        J->name = "J";

        T t = 0.0;

        int num_steps = 3 * num_phases * steps_per_phase;
        std::vector<T> losses(num_steps);

        for (int i = 0; i < num_steps; i++)
        {
            // Update current density
            update_current_density(J0z, omega_J, t, ct, currents);
            J->sub(2).interpolate(*J0z);

            // out.write_function(*J, t);

            // Update RHS
            VecSet(b, 0.0);
            assemble_vector_nest<T>(b, L, a_diag, bcs);
            copy(b, b_); // Copy b to b_

            VecSet(x_, 0.0);
            KSPSolve(solver, b_, x_);
            // Get number of iterations
            PetscInt iters;
            KSPGetIterationNumber(solver, &iters);

            // Get convergence reason
            KSPConvergedReason reason;
            KSPGetConvergedReason(solver, &reason);

            {
                // Update u_A
                copy(x_, x); // copy x_ to x

                // x = [x0, x1];
                Vec x0;
                VecNestGetSubVec(x, 0, &x0);
                copy<T>(x0, *u_A->x());  // copy x0 to u_A
                u_A->x()->scatter_fwd(); // Update ghost values

                // Update u_V
                Vec x1;
                VecNestGetSubVec(x, 1, &x1);
                copy<T>(x1, *u_V->x());  //  copy x1 to u_V
                u_V->x()->scatter_fwd(); // Update ghost values
            }

            T loss = fem::assemble_scalar(*LossForm);
            MPI_Allreduce(MPI_IN_PLACE, &loss, 1, MPI_DOUBLE, MPI_SUM, comm);
            losses[i] = loss;
            PetscPrintf(comm, "Loss: %f\n", loss);

            // Update previous solution - u_A0 = u_A
            // Copy all values from u_A to u_A0 including ghost values
            std::copy(u_A->x()->array().begin(), u_A->x()->array().end(), u_A0->x()->mutable_array().begin());

            // Update bfield
            B->interpolate(bfield_expr);
            out.write_function(*B, t);

            // Update time
            t += dt->value[0];
        }
        out.close();

        // Sum losses
        T total_loss = std::accumulate(losses.begin(), losses.end(), 0.0);
        PetscPrintf(comm, "Total loss: %f\n", (total_loss / t) * T_);
        // print T_
        PetscPrintf(comm, "T: %f\n", T_);
        // print t
        PetscPrintf(comm, "t: %f\n", t);
    }

    PetscFinalize();
    return 0;
}