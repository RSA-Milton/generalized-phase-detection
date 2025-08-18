import h5py

with h5py.File('model_pol_best.hdf5', 'r') as f:
    sequential_weights = f['model_weights']['sequential_1']
    print("Capas disponibles en HDF5:")
    for layer_name in sorted(sequential_weights.keys()):
        print(f"  - {layer_name}")