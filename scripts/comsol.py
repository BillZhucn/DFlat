import numpy as np
import mph
from dflat.metasurface import export_geometry_to_comsol

path = "./data/"

model_name = "Nanocylinders_TiO2_U300H600"

p = np.load(path+"designed_metasurface.npy")

print(p.shape, p.min(), p.max())

print("\nStarting COMSOL export...\n")

server = mph.Server(port=11451, arguments=["-Xmx48g"], version="6.4")
mph_client = None

print("Starting COMSOL server version "+server.version+"...\n")

mph_client = mph.Client(host="localhost", port=server.port, version=server.version)
mph_model = mph_client.create("metasurface")

print("COMSOL server started.\n")

export_geometry_to_comsol(
    p,
    model_name,
    mph_model=mph_model,
    debug=True,
    save_path=path+"metasurface-"+server.version+".mph"
    )

print("\nCOMSOL export complete.\n")

server.stop()

print("COMSOL server stopped.\nbye!")