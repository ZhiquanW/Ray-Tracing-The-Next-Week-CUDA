//
// Created by mySab on 2018/9/22.
//

#ifndef RAY_TRACING_ENGINE_SPHERE_H
#define RAY_TRACING_ENGINE_SPHERE_H

#include "hitinfo.cuh"
#include "material.cuh"
#include "object.cuh"
#include "stdio.h"
class sphere : public object {
private:
  vector3 center;
  float radius;
  material *material_ptr;
  vector3 center_start, center_end;

public:
  sphere() = default;

  __host__ __device__ sphere(const vector3 &_c, const float &_r, material *);

  __device__ bool hit(const ray &, const float &, const float &,
                      hitinfo &) const override;

  // bool displacement(const float &_time, vector3 &_target_pos) const override;
  __device__ vector3 center(flaot _t) const;

  __host__ void disp_info() const override;
};
__host__ __device__ sphere::sphere(const vector3 &_c, const float &_r,
                                   material *_m, const float &_t0 = 0,
                                   const float &_t1)
    : center(_c), radius(_r), material_ptr(_m) {}
__device__ bool sphere::hit(const ray &_r, const float &_min, const float &_max,
                            hitinfo &_info) const {
  vector3 center_t = this->center;
  // if (!this->displacement(_r.get_time(), center_t)) {
  //   center_t = this->center;
  // }
  vector3 oc = _r.origin() - center_t;
  float a = dot(_r.direction(), _r.direction());
  float b = dot(oc, _r.direction());
  float c = dot(oc, oc) - this->radius * this->radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0.0) {
    float tmp = (-b - sqrt(discriminant)) / a;
    if (tmp > _min && tmp < _max) {
      _info.dis = tmp;
      _info.pos = _r.target_pos(tmp);
      _info.normal = (_info.pos - center_t) / this->radius;
      _info.material_ptr = material_ptr;
      //_info.material_ptr = this->material_ptr;
      return true;
    }
    tmp = (-b + sqrt(discriminant)) / a;
    if (tmp > _min && tmp < _max) {
      _info.dis = tmp;
      _info.pos = _r.target_pos(_info.dis);
      _info.normal = (_info.pos - center_t) / this->radius;
      _info.material_ptr = material_ptr;
      // _info.material_ptr = this->material_ptr;
      return true;
    }
  }
  return false;
}

__device__ vector3 center(float _t) const {
  return center_start + (_t - time_start) / (time_end - time_start) *
                            (center_end - center_start);
}
__host__ void sphere::disp_info() const {}
#endif // RAY_TRACING_ENGINE_SPHERE_H
