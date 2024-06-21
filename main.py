import astra
import foam_ct_phantom as fcp
import h5py
import numpy as np
import tempfile
import time


def del_dataset_if_exists(file: h5py.File, dataset: str) -> None:
    if dataset in file:
        del file[dataset]


def gen_spheres(temp_file: str, z_max: float, spheres: int, trials: int, seed: int) -> np.ndarray:
    fcp.FoamPhantom.generate(
        temp_file,
        seed,
        nspheres_per_unit=spheres,
        ntrials_per_unit=trials,
        zrange=z_max,
        maxsize=0.2,
    )
    with h5py.File(temp_file, "r") as f:
        data = f["spheres"][()]
    return data


def gen_phantom(filename: str, spheres: int, trials: int | None = None, seed: int | None = None) -> None:
    z_max = 1.5
    if trials is None:
        trials = spheres * 10
    if seed is None:
        seed = int(time.time())
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.close()
        data = gen_spheres(temp.name, z_max, int(spheres / z_max + 0.5), int(trials / z_max + 0.5), seed)
    with h5py.File(filename, "w") as f:
        f.create_dataset("spheres", data=data, compression="gzip")
        f["spheres"].attrs["z_max"] = z_max
        f["spheres"].attrs["spheres"] = spheres
        f["spheres"].attrs["trials"] = trials
        f["spheres"].attrs["seed"] = seed


def copy_spheres(src: str, dest: str) -> None:
    with h5py.File(dest, "w") as f_dest:
        with h5py.File(src, "r") as f_src:
            f_dest.create_dataset("spheres", data=f_src["spheres"][()], compression="gzip")
            for name, value in f_src["spheres"].attrs.items():
                f_dest["spheres"].attrs[name] = value


def gen_volume(filename: str, size: int) -> None:
    with h5py.File(filename, "r") as f:
        z_max = f["spheres"].attrs["z_max"]
    phantom = fcp.FoamPhantom(filename)
    voxel_size = 2*z_max / size
    geom = fcp.VolumeGeometry(size, size, size, voxel_size)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.close()
        phantom.generate_volume(temp.name, geom)
        with h5py.File(temp.name, "r") as f:
            volume = f["volume"][()]
    with h5py.File(filename, "a") as f:
        del_dataset_if_exists(f, "volume")
        f.create_dataset("volume", data=volume, compression="gzip")
        f["volume"].attrs["size"] = size
        f["volume"].attrs["voxel_size"] = voxel_size


def gen_projs_accurate(filename: str, size: int, angles_num: int) -> None:
    obj = fcp.FoamPhantom(filename)
    with h5py.File(filename, "r") as f:
        z_max = f["spheres"].attrs["z_max"]
    pixel_size = 2*z_max / size
    angles = np.linspace(0, np.pi, num=angles_num, endpoint=False)
    geom = fcp.ParallelGeometry(size, size, angles, pixel_size)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.close()
        obj.generate_projections(temp.name, geom)
        with h5py.File(temp.name, "r") as f:
            projs = f["projs"][()]
    with h5py.File(filename, "a") as f:
        del_dataset_if_exists(f, "projs_accurate")
        f.create_dataset("projs_accurate", data=projs.swapaxes(0, 1), compression="gzip")
        f["projs_accurate"].attrs["size"] = size
        f["projs_accurate"].attrs["pixel_size"] = pixel_size
        f["projs_accurate"].attrs["angles_num"] = angles_num


def gen_projs_voxel(filename: str, angles_num: int) -> None:
    with h5py.File(filename, "r") as f:
        voxels = f["volume"][()]
    shape = voxels.shape
    max_size = max(shape)

    vol_geom = astra.creators.create_vol_geom(shape[1], shape[2], shape[0])
    obj_id = astra.data3d.create("-vol", vol_geom, data=voxels)

    angles = np.linspace(0, np.pi, num=angles_num, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel3d", 1.0, 1.0, max_size, max_size, angles)
    proj_id, proj = astra.creators.create_sino3d_gpu(obj_id, proj_geom, vol_geom)

    astra.data3d.delete(proj_id)
    astra.data3d.delete(obj_id)

    with h5py.File(filename, "a") as f:
        del_dataset_if_exists(f, "projs_voxel")
        f.create_dataset("projs_voxel", data=proj, compression="gzip")
        f["projs_voxel"].attrs["detector_spacing_x"] = 1.0
        f["projs_voxel"].attrs["detector_spacing_y"] = 1.0
        f["projs_voxel"].attrs["det_row_count"] = max_size
        f["projs_voxel"].attrs["det_col_count"] = max_size
        f["projs_voxel"].attrs["shape"] = shape
        f["projs_voxel"].attrs["angles_num"] = angles_num


def reconstruct_fbp(filename: str, dataset_in: str, dataset_out: str) -> None:
    with h5py.File(filename, "r") as f:
        projs = f[dataset_in][()]
        angles_num = f[dataset_in].attrs["angles_num"]
    shape = projs.shape
    angles = np.linspace(0, np.pi, num=angles_num, endpoint=False)

    proj_geom = astra.create_proj_geom("parallel", 1.0, shape[2], angles)
    vol_geom = astra.creators.create_vol_geom(shape[2], shape[2])

    rec = np.zeros((shape[0], shape[2], shape[2]))
    for i in range(shape[0]):
        proj_id = astra.data2d.create("-sino", proj_geom, projs[i])
        rec_id = astra.data2d.create("-vol", vol_geom)

        cfg = astra.astra_dict("FBP_CUDA")
        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectionDataId"] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        rec[i] = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

    with h5py.File(filename, "a") as f:
        del_dataset_if_exists(f, dataset_out)
        f.create_dataset(dataset_out, data=rec, compression="gzip")


def main() -> None:
    pass


if __name__ == "__main__":
    main()
