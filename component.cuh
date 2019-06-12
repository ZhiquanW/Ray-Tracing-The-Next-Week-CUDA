// created by zhiquan wang on 2019/06/12

#ifndef COMPONENT_CUH_CUDA
#define COMPONENT_CUH_CUDA
class component {
public:
  bool is_enable;
  __host__ __device__ component() : is_enable(false) {}
};
#endif