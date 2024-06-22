import click
import functions


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("filename", type=click.Path())
@click.argument("spheres", type=click.IntRange(min=1))
@click.option("--seed", type=click.IntRange(min=1), default=None)
def gen_phantom(filename: str, spheres: int, seed: int | None) -> None:
    functions.gen_phantom(filename, spheres, seed=seed)


@main.command()
@click.argument("src", type=click.Path())
@click.argument("dest", type=click.Path())
def copy_spheres(src: str, dest: str) -> None:
    functions.copy_spheres(src, dest)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("size", type=click.IntRange(min=8))
def gen_volume(filename: str, size: int) -> None:
    functions.gen_volume(filename, size)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("size", type=click.IntRange(min=8))
@click.option("--angles", type=click.IntRange(min=1), default=180)
def gen_projs_accurate(filename: str, size: int, angles: int) -> None:
    functions.gen_projs_accurate(filename, size, angles)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--angles", type=click.IntRange(min=1), default=180)
def gen_projs_voxel(filename: str, angles: int) -> None:
    functions.gen_projs_voxel(filename, angles)


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("dataset_in", type=str)
@click.argument("dataset_out", type=str)
def reconstruct_fbp(filename: str, dataset_in: str, dataset_out: str) -> None:
    # todo: not showing in git-bash
    with click.progressbar(length=functions.get_height(filename, dataset_in)) as bar:
        functions.reconstruct_fbp(filename, dataset_in, dataset_out, lambda: bar.update(1))


if __name__ == "__main__":
    main()
