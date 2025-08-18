import h5py

with h5py.File('model_pol_best.hdf5', 'r') as f:
    print("Claves en raíz:", list(f.keys()))
    if 'model_weights' in f:
        print("Contenido de model_weights:", list(f['model_weights'].keys()))