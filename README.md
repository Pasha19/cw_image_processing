# Install

## Install conda build

```
conda install conda-build
```

## Build

cd to source root

```
conda build . -c conda-forge -c astra-toolbox -c defaults
```

## Create env

```
conda create --name cadia python=3.10
```

## Install

```
conda activate cadia
# or
# source activate cadia
conda install --use-local cadia -c conda-forge -c astra-toolbox -c defaults
```
