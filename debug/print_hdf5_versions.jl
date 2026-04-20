
using VeryDiff.HDF5 

py_path = joinpath(@__DIR__, "..", ".CondaPkg", ".pixi", "envs", "default", "bin", "python")

println("--- Julia ---")
println("HDF5 library version: ", HDF5.libversion)
println()
println("--- Python ---")
run(`$py_path -c "import h5py; print('h5py version:', h5py.__version__); print('HDF5 library version:', h5py.version.hdf5_version)"`)


