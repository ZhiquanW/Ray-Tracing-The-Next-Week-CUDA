//
// Created by ZhiquanWang on 2018/7/12.
//

#ifndef RAY_TRACING_ENGINE_RAY_H
#define RAY_TRACING_ENGINE_RAY_H
#include "vector3.cuh"

class ray {
private:
public:
  vector3 pos;
  vector3 dir;
  double time;
  ray() = default;
  __device__ ray(double _p0, double _p1, double _p2, double _d0, double _d1,
                 double _d2, double _time) {
    pos.set_vec(_p0, _p1, _p2);
    dir.set_vec(_d0, _d1, _d2);
    this->dir = this->dir.normalize();
    this->time = _time;
  }
  __device__ ray(const vector3 &_p, const vector3 &_d, const double &_time = 0)
      : time(_time) {
    pos = _p;
    this->dir = _d.normalize();
  }

  __device__ vector3 origin() const { return pos; }

  __device__ vector3 direction() const { return dir; }

  __device__ vector3 target_pos(double _t) const { return pos + _t * dir; }

  __device__ double get_time() const { return this->time; }
};

#endif // RAY_TRACING_ENGINE_RAY_H
