//
// Created by ZhiquanWang on 2018/7/4.
//

#ifndef MATRIXOPERATION_CPP_vector3_H
#define MATRIXOPERATION_CPP_vector3_H

#include <cmath>
#include <curand_kernel.h>
#include <iostream>

class vector3 {
private:
  float vec3[3]{};

public:
  vector3() = default;

  __host__ __device__ vector3(const vector3 &);

  __host__ __device__ vector3(float _n0, float _n1, float _n2);

  __host__ __device__ inline void set_vec(float, float, float);

  __host__ __device__ inline float x() const;

  __host__ __device__ inline float y() const;

  __host__ __device__ inline float z() const;

  __host__ __device__ inline float r() const;

  __host__ __device__ inline float g() const;

  __host__ __device__ inline float b() const;

  __host__ __device__ inline const vector3 &operator+() const;

  __host__ __device__ inline vector3 operator-();

  __host__ __device__ inline float operator[](int) const;

  __host__ __device__ inline float &operator[](int);

  __host__ __device__ inline vector3 &operator+=(const vector3 &);

  __host__ __device__ inline vector3 &operator-=(const vector3 &);

  __host__ __device__ inline vector3 &operator*=(const vector3 &);

  __host__ __device__ inline vector3 &operator*=(const float &);

  __host__ __device__ inline vector3 &operator/=(const vector3 &);

  __host__ __device__ inline vector3 &operator/=(const float &);

  __host__ __device__ inline bool operator==(const vector3 &);

  __host__ __device__ inline bool operator!=(const vector3 &);

  __host__ __device__ inline float length() const;

  __host__ __device__ inline float squared_length() const;

  __host__ __device__ inline vector3 normalize() const;
};

__host__ __device__ inline void vector3::set_vec(float _p0, float _p1,
                                                 float _p2) {
  vec3[0] = _p0;
  vec3[1] = _p1;
  vec3[2] = _p2;
}
__host__ __device__ vector3::vector3(const vector3 &_v) {
  for (int i = 0; i < 3; ++i) {
    this->vec3[i] = _v[i];
  }
}

__host__ __device__ vector3::vector3(float _n0, float _n1, float _n2) {
  this->vec3[0] = _n0;
  this->vec3[1] = _n1;
  this->vec3[2] = _n2;
}

__host__ __device__ inline float vector3::x() const { return this->vec3[0]; }

__host__ __device__ inline float vector3::y() const { return this->vec3[1]; }

__host__ __device__ inline float vector3::z() const { return this->vec3[2]; }

__host__ __device__ inline float vector3::r() const { return this->vec3[0]; }

__host__ __device__ inline float vector3::g() const { return this->vec3[1]; }

__host__ __device__ inline float vector3::b() const { return this->vec3[2]; }

__host__ __device__ inline const vector3 &vector3::operator+() const {
  return *this;
}

__host__ __device__ inline vector3 vector3::operator-() {
  return vector3(-1 * this->vec3[0], -1 * this->vec3[1], -1 * this->vec3[2]);
}

__host__ __device__ inline float vector3::operator[](int _i) const {
  return this->vec3[_i];
}

__host__ __device__ inline float &vector3::operator[](int _i) {
  return this->vec3[_i];
}

__host__ __device__ inline vector3 &vector3::operator+=(const vector3 &_v) {
  for (int i = 0; i < 3; ++i) {
    this->vec3[i] += _v[i];
  }
  return *this;
}

__host__ __device__ inline vector3 &vector3::operator-=(const vector3 &_v) {
  for (int i = 0; i < 3; ++i) {
    this->vec3[i] -= _v[i];
  }
  return *this;
}

__host__ __device__ inline vector3 &vector3::operator*=(const vector3 &_v) {
  for (int i = 0; i < 3; ++i) {
    this->vec3[i] *= _v[i];
  }
  return *this;
}

__host__ __device__ inline vector3 &vector3::operator*=(const float &_d) {
  for (float &data : this->vec3) {
    data *= _d;
  }
  return *this;
}

__host__ __device__ inline vector3 &vector3::operator/=(const vector3 &_v) {
  for (int i = 0; i < 3; ++i) {
    this->vec3[i] /= _v[i];
  }
  return *this;
}

__host__ __device__ inline vector3 &vector3::operator/=(const float &_d) {
  for (float &data : this->vec3) {
    data /= _d;
  }
  return *this;
}

__host__ __device__ inline float vector3::length() const {
  return sqrt(this->vec3[0] * this->vec3[0] + this->vec3[1] * this->vec3[1] +
              this->vec3[2] * this->vec3[2]);
}
__host__ __device__ inline float vector3::squared_length() const {
  return this->vec3[0] * this->vec3[0] + this->vec3[1] * this->vec3[1] +
         this->vec3[2] * this->vec3[2];
}

__host__ __device__ inline vector3 vector3::normalize() const {
  float tmp_len = this->length();
  vector3 tmp_v = *this;
  for (float &data : tmp_v.vec3) {
    data /= tmp_len;
  }
  return tmp_v;
}

__host__ __device__ inline bool vector3::operator==(const vector3 &_v) {
  for (int i = 0; i < 3; ++i) {
    if (this->vec3[i] != _v[i]) {
      return false;
    }
  }
  return true;
}

__host__ __device__ inline bool vector3::operator!=(const vector3 &_v) {
  return !(*this == _v);
}

inline std::istream &operator>>(std::istream &_is, vector3 &_v) {
  _is >> _v[0] >> _v[1] >> _v[2];
  return _is;
}

inline std::ostream &operator<<(std::ostream &_os, const vector3 &_v) {
  _os << "(" << _v[0] << "," << _v[1] << "," << _v[2] << ")";
  return _os;
}

__host__ __device__ inline float dot(const vector3 &_v0, const vector3 &_v1) {
  float tmp_result = 0;
  for (int i = 0; i < 3; ++i) {
    tmp_result += _v0[i] * _v1[i];
  }
  return tmp_result;
}

__host__ __device__ inline vector3 cross(const vector3 &_v0,
                                         const vector3 &_v1) {
  return {_v0[1] * _v1[2] - _v0[2] * _v1[1], _v0[2] * _v1[0] - _v0[0] * _v1[2],
          _v0[0] * _v1[1] - _v0[1] * _v1[0]};
}

__host__ __device__ inline vector3 operator+(const vector3 &_v0,
                                             const vector3 &_v1) {
  return {_v0[0] + _v1[0], _v0[1] + _v1[1], _v0[2] + _v1[2]};
}

__host__ __device__ inline vector3 operator-(const vector3 &_v0,
                                             const vector3 &_v1) {
  return {_v0[0] - _v1[0], _v0[1] - _v1[1], _v0[2] - _v1[2]};
}

__host__ __device__ inline vector3 operator*(const vector3 &_v0,
                                             const vector3 &_v1) {
  return {_v0[0] * _v1[0], _v0[1] * _v1[1], _v0[2] * _v1[2]};
}

__host__ __device__ inline vector3 operator*(const vector3 &_v0,
                                             const float &_d) {
  return {_v0[0] * _d, _v0[1] * _d, _v0[2] * _d};
}

__host__ __device__ inline vector3 operator*(const float &_d,
                                             const vector3 &_v0) {
  return {_v0[0] * _d, _v0[1] * _d, _v0[2] * _d};
}

__host__ __device__ inline vector3 operator/(const vector3 &_v0,
                                             const vector3 &_v1) {
  return {_v0[0] / _v1[0], _v0[1] / _v1[1], _v0[2] / _v1[2]};
}

__host__ __device__ inline vector3 operator/(const vector3 &_v0,
                                             const float &_d) {
  return {_v0[0] / _d, _v0[1] / _d, _v0[2] / _d};
}

#endif // MATRIXOPERATION_CPP_vector3_H
