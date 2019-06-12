//
// Created by mySab on 2018/9/22.
//

#ifndef RAY_TRACING_ENGINE_SPHERE_H
#define RAY_TRACING_ENGINE_SPHERE_H

#include "hitinfo.cuh"
#include "material.cuh"
#include "movement.cuh"
#include "object.cuh"
#include "stdio.h"
class sphere : public object, public movement {
private:
  vector3 center_pos;
  float radius;
  material *material_ptr;
  movement *movement_ptr;

public:
  sphere() = default;

  __host__ __device__ sphere(const vector3 &_c, const float &_r, material *,
                             movement * = nullptr);

  __device__ bool hit(const ray &, const float &, const float &,
                      hitinfo &) const override;

  // bool displacement(const float &_time, vector3 &_target_pos) const override;
  __device__ vector3 center(float _t) const;

  __host__ void disp_info() const override;
};
__host__ __device__ sphere::sphere(const vector3 &_c, const float &_r,
                                   material *_m, movement *_mv)
    : center_pos(_c), radius(_r), material_ptr(_m), movement_ptr(_mv) {}
__device__ bool sphere::hit(const ray &_r, const float &_min, const float &_max,
                            hitinfo &_info) const {
  vector3 center_t = center_pos;
  if (movement_ptr->is_enable) {
    center_t = this->center(_r.time());
  }

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
      return true;
    }
    tmp = (-b + sqrt(discriminant)) / a;
    if (tmp > _min && tmp < _max) {
      _info.dis = tmp;
      _info.pos = _r.target_pos(_info.dis);
      _info.normal = (_info.pos - center_t) / this->radius;
      _info.material_ptr = material_ptr;
      return true;
    }
  }
  return false;
}

__device__ vector3 sphere::center(float _t) const {
  float move_duration =
      _t - movement_ptr->start_time() > movement_ptr->time_frame()
          ? movement_ptr->time_frame()
          : _t - movement_ptr->start_time();
  return center_pos + move_duration * movement_ptr->velocity();
  ;
}
__host__ void sphere::disp_info() const {}
#endif // RAY_TRACING_ENGINE_SPHERE_H
