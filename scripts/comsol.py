import numpy as np
import mph
from dflat.metasurface import export_geometry_to_comsol

path = "./data/"

model_name = "Nanocylinders_TiO2_U300H600"

p = np.load(path+"designed_metasurface.npy")

print(p.shape, p.min(), p.max())

print("\nStarting COMSOL export...\n")

mph.option('classkit', False)

mph_client = None

version = '6.3'
port = 11451

mph_client = mph.Client(host="localhost", port=port, version=version)
mph_model = mph_client.load(path+"metasurface-6.3.mph")

print("COMSOL server connected.\n")

export_geometry_to_comsol(
    p,
    model_name,
    mph_model=mph_model,
    debug=False,
    save_path=path+"metasurface-6.3.mph"
    )

print("\nCOMSOL export complete.\n")

server.stop()

print("COMSOL server stopped.\nbye!")