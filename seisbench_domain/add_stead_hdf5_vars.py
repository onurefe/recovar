import h5py

stead_waveforms_hdf5 = "/home/onur/stead/waveforms.hdf5"

with h5py.File(stead_waveforms_hdf5, "a") as f:
    f["data_format/dimension_order"] = "CW"
    f["data_format/component_order"] = "ENZ"
    f["data_format/sampling_rate"] = 100.0
