// Created by Zhiquan Wang on 2019/06/11
#ifndef MOVEMENT_CUH_CUDA
#define MOVEMENT_CUH_CUDA

#include "component.cuh"
#include "vector3.cuh"
class movement : public component {
public:
  float time_list[2];
  vector3 obj_velocity;
  __host__ __device__ movement() : obj_velocity(vector3(0, 0, 0)) {
    time_list[0] = 0;
    time_list[1] = 0;
  }
  __host__ __device__ movement(const float &_st, const float &_tf,
                               const vector3 &_v)
      : component(), obj_velocity(_v) {
    time_list[0] = _st;
    time_list[1] = _tf;
    this->is_enable = true;
  }
  __host__ __device__ float start_time() { return time_list[0]; }
  __host__ __device__ float time_frame() { return time_list[1]; }
  __host__ __device__ vector3 velocity() const { return obj_velocity; }
};
#endif