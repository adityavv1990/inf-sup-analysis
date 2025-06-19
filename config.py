eigenSolver = 'petsc'  # 'petsc' or 'scipy'
if eigenSolver not in ['petsc', 'scipy']:
    raise ValueError("Invalid eigenSolver option. Choose 'petsc' or 'scipy'.")

formulation = 'mixed' # 'mixed' or 'standard'
if formulation not in ['mixed', 'standard']:
    raise ValueError("Invalid formulation option. Choose 'mixed' or 'standard'.")