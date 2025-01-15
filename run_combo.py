import os
import numpy as np

from file_utils import replace_text, delete_folder


def run_combo_simulation_list(combinations, iris_original, folder_path, input_iris_names):
    """
    Run simulation experiments with different combinations of input parameters.

    Parameters
    ----------
    combinations : np.ndarray
        An array containing combinations of input values for each parameter.
    iris_original : str
        The file path of the original 'main_reference.iris' file.
    folder_path : str
        The base folder path where the experiments will be conducted.
    input_iris_names : np.ndarray
        An array containing names for the input parameters.

    Returns
    -------
    None

    Notes
    -----
    This function iterates through all combinations of input arrays and runs simulations
    with different parameter values. It creates directories for each combination and
    updates the main.iris file within each directory using the specified input names
    and values. The simulations are then executed using the 'iris' command.

    Parameters
    ----------
    combinations : np.ndarray
        An array containing combinations of input values for each parameter.
    iris_original : str
        The file path of the original 'main_reference.iris' file.
    folder_path : str
        The base folder path where the experiments will be conducted.
    input_iris_names : np.ndarray
        An array containing names for the input parameters.

    Returns
    -------
    None

    Examples
    --------
    >>> iris_original = '/path/to/main_reference.iris'
    >>> folder_path = '/path/to/experiments'
    >>> inputs_array = [param1_values, param2_values, ...]
    >>> input_iris_names = ['param1', 'param2', ...]
    >>> run_combo_simulation_list(inputs_array, iris_original, folder_path, input_iris_names)
    """
    # Create a structured array with a dtype that includes index and combinations
    dtypes = [(f"{name}", combinations.dtype) for name in input_iris_names]
    dtypes.insert(0, ('Index', np.int))

    structured_array = np.zeros(combinations.shape[0], dtype=dtypes)

    for i, name in enumerate(input_iris_names):
        structured_array[name] = combinations[:, i]

    structured_array['Index'] = np.arange(combinations.shape[0])

    # Save the structured array to a text file
    simulations_folder = os.path.join(folder_path, "combo_simulations")
    delete_folder(simulations_folder)
    os.mkdir(simulations_folder)
    output_file_path = os.path.join(simulations_folder, 'combinations_info.txt')
    np.savetxt(output_file_path, structured_array, delimiter='\t', fmt='%d\t' + '%s\t' *
               (len(input_iris_names) - 1) + '%s', header='\t'.join(structured_array.dtype.names), comments='')

    # Iterate through all combinations of input arrays
    simu_number = 0
    for combination in combinations:

        simu_folder = os.path.join(simulations_folder, str(simu_number))
        os.mkdir(simu_folder)

        os.chdir(simu_folder)
        ################################################################
        # search_text      replace_text
        list_replace = [
            [input_iris_names[i], combination[i]] for i in range(len(combination))
        ]

        target_iris = os.path.join(simu_folder, "main.iris")
        replace_text(iris_original, target_iris, list_replace)

        os.system('iris')
        os.chdir("../")
        simu_number += 1


def run_combo_simulation(iris_original, folder_path, inputs_array, input_iris_names):
    """
    Run simulation experiments with different combinations of input parameters.

    Parameters
    ----------
    iris_original : str
        The file path of the original 'main_reference.iris' file.
    folder_path : str
        The base folder path where the experiments will be conducted.
    inputs_array : np.ndarray
        An array containing input values for each parameter.
    input_iris_names : np.ndarray
        An array containing names for the input parameters.

    Returns
    -------
    None

    Notes
    -----
    This function iterates through all combinations of input arrays and runs simulations
    with different parameter values. It creates directories for each combination and
    updates the main.iris file within each directory using the specified input names
    and values. The simulations are then executed using the 'iris' command.

    Examples
    --------
    >>> iris_original = '/path/to/main_reference.iris'
    >>> folder_path = '/path/to/experiments'
    >>> inputs_array = [param1_values, param2_values, ...]
    >>> input_iris_names = ['param1', 'param2', ...]
    >>> run_combo_simulation(iris_original, folder_path, inputs_array, input_iris_names)
    """
    # Iterate through all combinations of input arrays
    for combination in np.array(np.meshgrid(*inputs_array)).T.reshape(-1, len(inputs_array)):
        directories = [folder_path]

        for i, value in enumerate(combination):
            directory_i = os.path.join(directories[-1], value)
            if not os.path.exists(directory_i):
                os.mkdir(directory_i)
            directories.append(directory_i)

        ################################################################
        # search_text      replace_text
        list_replace = [
            [input_iris_names[i], combination[i]] for i in range(len(combination))
        ]

        print(directories[-1])
        target_iris = os.path.join(directories[-1], "main.iris")
        replace_text(iris_original, target_iris, list_replace)

        os.chdir(directories[-1])
        os.system('iris')
        os.chdir("../" * len(directories[:-1]))
