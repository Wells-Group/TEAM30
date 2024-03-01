#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/petsc.h>

#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/common/IndexMap.h>
#include <memory>

#include "utils.hpp"

using namespace dolfinx;

// NLS
class NonlinearEMProblem
{
public:
    NonlinearEMProblem(
        std::vector<std::shared_ptr<fem::Form<T>>> L, std::vector<std::shared_ptr<fem::Form<T>>> J,
        std::vector<std::shared_ptr<const fem::DirichletBC<T>>> bcs)
        : _l(L), _j(J), _bcs(bcs),
          _b(L->function_spaces()[0]->dofmap()->index_map,
             L->function_spaces()[0]->dofmap()->index_map_bs()),
          _matA(la::petsc::Matrix(fem::petsc::create_matrix(*J, "aij"), false))
    // _matA(la::petsc::Matrix(fem::petsc::create_matrix(*J, "baij"), false))
    {
        // Maps for the vector
        std::vector<
            std::pair<std::reference_wrapper<const common::IndexMap>, int>>
            maps;
        for (const auto &form : L)
        {
            auto map = form->function_spaces()[0]->dofmap()->index_map;
            int bs = form->function_spaces()[0]->dofmap()->index_map_bs();
            maps.push_back({*map, bs});
        }

        // Create the vector
        _b_petsc = fem::petsc::create_vector_nest(maps);
    }

    /// Destructor
    virtual ~NonlinearEMProblem()
    {
        if (_b_petsc)
            VecDestroy(&_b_petsc);
    }

    auto form()
    {
        return [](Vec x)
        {
            // Get number of vested Vecs
            PetscInt num_vecs;
            VecNestGetSize(x, &num_vecs);

            // Assemble each block of the vector
            for (PetscInt idxm = 0; idxm < num_vecs; idxm++)
            {
                Vec xi;
                VecNestGetSubVec(b, idxm, &xi);
                VecGhostUpdateBegin(xi, INSERT_VALUES, SCATTER_FORWARD);
                VecGhostUpdateEnd(xi, INSERT_VALUES, SCATTER_FORWARD);
            }
        };
    }

    /// Compute F at current point x
    auto F()
    {
        return [&](const Vec x, Vec)
        {
            // Assemble b and update ghosts
            assemble_vector_nest(_b_petsc, L, a, bcs, x, 1.0);
        };
    }

    /// Compute J = F' at current point x
    auto J()
    {
        return [&](const Vec, Mat A)
        {
            std::array<PetscInt, 2> shape;
            shape[0] = _j.size();
            shape[1] = _j.front.size();

            // Assemble each block of the matrix
            for (PetscInt idxm = 0; idxm < shape[0]; idxm++)
            {
                for (PetscInt jdxm = 0; jdxm < shape[1]; jdxm++)
                {
                    Mat Aij;
                    MatNestGetSubMat(A.mat(), idxm, jdxm, &Aij);
                    MatSetOption(Aij, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
                    fem::assemble_matrix(la::petsc::Matrix::set_fn(Aij, ADD_VALUES), *_j[idxm][jdxm], bcs);
                    MatAssemblyBegin(Aij, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(Aij, MAT_FLUSH_ASSEMBLY);
                    if (idxm == jdxm)
                        fem::set_diagonal<T>(la::petsc::Matrix::set_fn(Aij, INSERT_VALUES), *_j[idxm][jdxm]->function_spaces()[0], bcs);
                    MatAssemblyBegin(Aij, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(Aij, MAT_FINAL_ASSEMBLY);
                }
            }
            MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
        };
    }

    Vec vector() { return _b_petsc; }

    Mat matrix() { return _matA.mat(); }

private:
    std::vector<std::shared_ptr<fem::Form<T>>> _l, _j;
    std::vector<std::shared_ptr<const fem::DirichletBC<T>>> _bcs;
    Vec _b_petsc = nullptr;
    la::petsc::Matrix _matA;
};
