//
// Created by Zhiquan on 2019/6/1.
//

#ifndef RAY_TRACING_ENGINE_OBJECT_CUDA_H
#define RAY_TRACING_ENGINE_OBJECT_CUDA_H

#include "hitinfo.cuh"
#include "ray.cuh"

class object {
public:
  __device__ virtual bool hit(const ray &, const float &, const float &,
                              hitinfo &) const = 0;
  __host__ virtual void disp_info() const = 0;
};

#endif // RAY_TRACING_ENGINE_OBJECT_CUDA_H
