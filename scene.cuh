//
// Created by mySab on 2019/06/01.
//

#ifndef RAY_TRACING_ENGINE_SCENE_CUDA_H
#define RAY_TRACING_ENGINE_SCENE_CUDA_H

#include "hitinfo.cuh"
#include "object.cuh"
#include "stdio.h"
#include <iostream>

class scene : public object {
public:
  object **object_list;
  unsigned int object_num;
  unsigned int list_len;

  scene() = default;
  __host__ __device__ scene(const unsigned int &);
  __host__ __device__ scene(object **, int);

  __host__ __device__ bool add_object(object *);
  __device__ bool hit(const ray &, const float &, const float &,
                      hitinfo &) const override;

  __host__ void disp_info() const override;

  __host__ __device__ void clear();
};

__host__ __device__ scene::scene(const unsigned int &_l_num)
    : object_num(0), list_len(_l_num) {
  object_list = new object *[_l_num];
}
__host__ __device__ scene::scene(object **_objs, int _len) {
  object_list = _objs;
  object_num = _len;
  list_len = _len;
}

__host__ __device__ bool scene::add_object(object *_tmp_o) {
  if (object_num < list_len) {
    object_list[object_num++] = _tmp_o;
    return true;
  }
  return false;
}

__device__ bool scene::hit(const ray &_r, const float &_min, const float &_max,
                           hitinfo &_info) const {
  float tmp_min = _max;
  hitinfo tmp_info;
  bool is_hit = false;
  for (int i = 0; i < list_len; ++i) {
    if (object_list[i]->hit(_r, _min, tmp_min, tmp_info)) {
      is_hit = true;
      tmp_min = tmp_info.dis;
      _info = tmp_info;
    }
  }
  return is_hit;
}

__host__ void scene::disp_info() const {
  std::cerr << "display scene info -> list_length : " << list_len
            << " -> object number: " << object_num << std::endl;
  for (int i = 0; i < list_len; ++i) {
    object_list[i]->disp_info();
  }
}
#endif // RAY_TRACING_ENGINE_SCENE_H
