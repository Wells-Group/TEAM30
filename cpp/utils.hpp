#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/MeshTags.h>

#include <cassert>

/// @brief Create discrete gradient matrix
/// @param V0 Function space to interpolate from
/// @param V1 Function space to interpolate to
/// @return Discrete gradient matrix
template <typename T, typename U>
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
template <typename T, typename U>
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
template <typename T>
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