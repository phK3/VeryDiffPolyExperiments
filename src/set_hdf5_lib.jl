
function set_hdf5_lib()
    lib_dir = joinpath(CondaPkg.envdir(), "lib")

    set_preferences!(
        HDF5,
        "libhdf5" => joinpath(lib_dir, "libhdf5.so"),
        "libhdf5_hl" => joinpath(lib_dir, "libhdf5_hl.so"),
        force=true
    )

    @info "HDF5.jl set to use HDF5 library at $(lib_dir)"
    @info "Restart Julia session after running this function to ensure changes take effect."
end