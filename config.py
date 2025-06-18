eigenSolver = 'petsc'  # 'petsc' or 'scipy'
if eigenSolver not in ['petsc', 'scipy']:
    raise ValueError("Invalid eigenSolver option. Choose 'petsc' or 'scipy'.")