import click
import functions


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("filename", type=click.Path())
@click.argument("spheres", type=click.IntRange(min=1))
@click.option("--seed", type=click.IntRange(min=1), default=None, help="Seed for random number generator")
def gen_phantom(filename: str, spheres: int, seed: int | None) -> None:
    """Generate phantom"""
    functions.gen_phantom(filename, spheres, seed=seed)


@main.command()
@click.argument("src", type=click.Path())
@click.argument("dest", type=click.Path())
def copy_spheres(src: str, dest: str) -> None:
    """Copy spheres from one h5 file to another"""
    functions.copy_spheres(src, dest)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("size", type=click.IntRange(min=8))
def gen_volume(filename: str, size: int) -> None:
    """Generate volume nxnxn from spheres dataset from h5 file"""
    functions.gen_volume(filename, size)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("size", type=click.IntRange(min=8))
@click.option("--angles", type=click.IntRange(min=1), default=180)
def gen_projs_accurate(filename: str, size: int, angles: int) -> None:
    """Genarate accurate projections from spheres dataset from h5 file"""
    functions.gen_projs_accurate(filename, size, angles)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--angles", type=click.IntRange(min=1), default=180)
def gen_projs_voxel(filename: str, angles: int) -> None:
    """Generate projections from voxel volume from spheres dataset from h5 file"""
    functions.gen_projs_voxel(filename, angles)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("dataset_in", type=str)
@click.argument("dataset_out", type=str)
def reconstruct_fbp(filename: str, dataset_in: str, dataset_out: str) -> None:
    """Reconstruct from projections from dataset from h5 file"""
    # todo: not showing in git-bash
    with click.progressbar(length=functions.get_height(filename, dataset_in)) as bar:
        functions.reconstruct_fbp(filename, dataset_in, dataset_out, lambda: bar.update(1))


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("dataset", type=str)
@click.argument("output", type=click.Path())
@click.option("--axis", type=click.IntRange(min=0, max=2))
@click.option("--ind", type=click.IntRange(min=0))
@click.option("--flip", is_flag=True, default=False, help="Flip x axis")
def img(filename: str, dataset: str, output: str, axis: int, ind: int, flip: bool) -> None:
    """Create image of slice of 3d volume"""
    functions.img(filename, dataset, output, axis, ind, flip)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--height", type=click.IntRange(min=0))
@click.option("--row", type=click.IntRange(min=0))
@click.option("--offset", type=click.IntRange(min=0), default=0, help="Offset from start and end of image")
@click.option("--diff", is_flag=True, default=False, help="Show difference with voxel volume")
def show_row(
        filename: str,
        output: str,
        height: int,
        row: int,
        offset: int,
        diff: bool,
) -> None:
    """Show single row of slice"""
    functions.show_row(
        filename,
        ("rec_fbp_accurate", "rec_fbp_voxel"),
        ("accurate", "voxel"),
        output,
        height,
        row,
        offset,
        diff,
    )


if __name__ == "__main__":
    main()
