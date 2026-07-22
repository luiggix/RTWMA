import os
import subprocess

def workingDirectoryFile(paths):
    """
    Creación del archivo "workingDirectory.txt" necesario para 
    la ejecución del programa "TR_1D_oper.exe"
    """
    filename = paths["wma_workingDirectory_file"]
    if not os.path.isfile(filename):
        print(f"-> File {filename} does not exist")
        print(f"-> Generating {filename}: ... \n")
        with open(filename, "w") as f:
            f.write(paths["wma_working_dir"])
    else:
        print(f"-> File {filename} already exist")


def run(paths):
    #
    # Checamos si existe el archivo #workingDirectory.txt"
    workingDirectoryFile(paths)
    #
    # Ejecutamos el programa
    result = subprocess.run([paths["tr1d_exe"]], 
                        cwd = paths["wma_working_dir"], 
                        capture_output = True, 
                        text = True)
    #
    # Salida de "TR_1D_oper.exe"
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)

