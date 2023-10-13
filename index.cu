#include <iostream>

__forceinline__ __device__ unsigned getLaneId()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned getWarpId()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__global__  void  index()
{
    auto warp = getWarpId();
    auto lane = getLaneId();

    printf("%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d, %d\n",blockIdx.y,blockIdx.x,gridDim.y,gridDim.x,warp,threadIdx.y,threadIdx.x,blockDim.y,blockDim.x);
}

int main(int argc,char *argv[])
{
    if (argc != 5)
    {   
        printf("Usage index [grid.x] [grid.y] [block.x] [block.y]");
        return 0;
    }
    unsigned int grid_x = std::strtoul(argv[1], nullptr, 10);
    unsigned int grid_y = std::strtoul(argv[2], nullptr, 10);
    unsigned int block_x = std::strtoul(argv[3], nullptr, 10);
    unsigned int block_y = std::strtoul(argv[4], nullptr, 10);

    index<<<{grid_x,grid_y},{block_x,block_y}>>>();
    cudaDeviceReset();
}