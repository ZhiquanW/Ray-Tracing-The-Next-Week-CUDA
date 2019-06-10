#ifndef HITINFO_CUDA_H
#define HITINFO_CUDA_H
#include "material.cuh"
#include "ray.cuh"
class material;
class hitinfo {
public:
  float dis;
  vector3 pos;
  vector3 normal;
  material *material_ptr;
  __host__ __device__ hitinfo() {}
};
#endif