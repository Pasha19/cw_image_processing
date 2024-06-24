import astra
import foam_ct_phantom as fcp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time

from PIL import Image
from typing import Callable


def del_dataset_if_exists(file: h5py.File, dataset: str) -> None:
    if dataset in file:
        del file[dataset]


def gen_phantom(filename: str, spheres: int, trials: int | None = None, seed: int | None = None) -> None:
    z_max = 1.5
    if trials is None:
        trials = spheres * 10
    if seed is None:
        seed = int(time.time())
    spheres = int(spheres / z_max + 0.5)
    trials = int(trials / z_max + 0.5)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.close()
        fcp.FoamPhantom.generate(
            temp.name,
            seed,
            nspheres_per_unit=spheres,
            ntrials_per_unit=trials,
            zrange=z_max,
            maxsize=0.2,
        )
        with h5py.File(temp.name, "r") as f:
            data = f["spheres"][()]
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


def get_height(filename: str, dataset_in: str) -> int:
    with h5py.File(filename, "r") as f:
        dataset = f[dataset_in]
        if "size" in dataset.attrs:
            return dataset.attrs["size"]
        return max(f[dataset_in].attrs["shape"])


def reconstruct_fbp(
        filename: str,
        dataset_in: str,
        dataset_out: str,
        iter_callback: Callable[[], None] | None = None
) -> None:
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

        if isinstance(iter_callback, Callable):
            iter_callback()

    with h5py.File(filename, "a") as f:
        del_dataset_if_exists(f, dataset_out)
        f.create_dataset(dataset_out, data=rec, compression="gzip")


def normalize(data: np.ndarray) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def save_img(data: np.ndarray, output: str) -> None:
    viridis_cmap = plt.get_cmap("viridis")
    norm_data = normalize(data)
    im = Image.fromarray((255 * viridis_cmap(norm_data)).astype(np.uint8))
    im.save(output)


def img(filename: str, dataset: str, output: str, axis: int, ind: int, flip: bool) -> None:
    with h5py.File(filename, "r") as f:
        data = f[dataset][()]
    if axis == 0:
        result = data[ind, :, :]
    elif axis == 1:
        result = data[:, ind, :]
    elif axis == 2:
        result = data[:, :, ind]
    else:
        raise ValueError(f"axis must be 0 or 1 or 2, got {axis}")
    save_img(np.flip(result, axis=0), output)


def show_row(
        filename: str,
        datasets: tuple[str, str],
        titles: tuple[str, str],
        output: str,
        height: int,
        row: int,
        offset: int = 0,
        diff: bool = False,
) -> tuple[float, float]:
    with h5py.File(filename, "r") as f:
        volume = f["volume"][()][height]
        size = f["volume"].attrs["size"]
        rec = f[datasets[0]][()][height], f[datasets[1]][()][height]
    norm_row = np.flip(normalize(rec[0]), axis=0), np.flip(normalize(rec[1]), axis=0)
    plt.figure(figsize=(8, 5))
    plt.xlim([offset, size - offset - 1])
    if diff:
        plt.plot(norm_row[0][row] - volume[row], label=titles[0])
        plt.plot(norm_row[1][row] - volume[row], label=titles[1])
    else:
        plt.plot(norm_row[0][row], label=titles[0])
        plt.plot(norm_row[1][row], label=titles[1])
    plt.legend(loc="upper right", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output)

    return np.squeeze(np.subtract(norm_row[0], volume)).mean(), np.squeeze(np.subtract(norm_row[1], volume)).mean()
