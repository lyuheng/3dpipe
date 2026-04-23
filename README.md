# 3DPipe: A Pipelined GPU Framework for Scalable Generalized Spatial Join over Polyhedral Objects

## Introduction
3DPipe is a pipelined GPU framework for scalable spatial join over polyhedral objects. 3DPipe exploits GPU parallelism across both filtering and refinement stages, incorporates a multi-level pruning strategy for efficient candidate reduction, and employs chunked streaming to handle datasets exceeding GPU memory. Its pipelined execution overlaps CPU data preparation, host-device data transfer, and GPU computation to improve throughput. 

More technical details can be found at: https://arxiv.org/abs/2604.19982.

## Dependencies
 - CMake 
 - NVCC
 - CGAL 
 - GMP 
 - Boost
 - OpenMP


## How to Compile Your Code
1. Set `CMAKE_CUDA_COMPILER` in src/CMakeLists.txt.
```cmake
set(CMAKE_CUDA_COMPILER /PATH/TO/CUDA/COMPILER)
```

2. Link path to `mpfr` in src/CMakeLists.txt.
```cmake
link_directories(/PATH/TO/mpfr_install/lib/)
```

3. Compile the code via CMake
```console
cd src/
cmake -B build
cmake --build build
```

## Generating Synthetic Dataset 
1. Generate two .dt files, one for *nuclei* (bar_n_nv1000_nu200_vs100_r30_cm1.dt) and one for *vessel* (bar_v_nv1000_nu200_vs100_r30_cm1.dt). 

- `nv` specifies the total number of vessels in the dataset.

- `nu` specifies the number of nuclei around *each* vessel. 

Therefore this generated dataset contains 1000 vessels and 200,000 nuclei
```console
./simulator -n ../../data/nuclei.pt -v ../../data/vessel.pt -o bar --nv 1000 --nu 200
```

2. Convert .dt files into decoded format for better performance but with higher storage cost.
```console
./tdbase convert bar_n_nv1000_nu200_vs100_r30.dt foo_n_nv1000_nu200_vs100_r30_cm1.dt
./tdbase convert bar_v_nv1000_nu200_vs100_r30.dt foo_v_nv1000_nu200_vs100_r30_cm1.dt
```

## Running on Generated Datasets

- Conduct a 10-NN join.
```console
./tdbase join --tile1 foo_n_nv1000_nu200_vs100_r30_cm1.dt --tile2 foo_v_nv1000_nu200_vs100_r30_cm1.dt -q nn --knn 10 -g --lod 20 40 60 80 100
```

- Conduct a within distance join with $\tau = 400$.
```console
./tdbase join --tile1 foo_n_nv1000_nu200_vs100_r30_cm1.dt --tile2 foo_v_nv1000_nu200_vs100_r30_cm1.dt -q within --within_dist 400 -g --lod 20 40 60 80 100 
```

- Conduct a 10-NN self join.
```console
./tdbase join --tile1 foo_n_nv1000_nu200_vs100_r30_cm1.dt -q nn --knn 10 -g --lod 20 40 60 80 100
```


