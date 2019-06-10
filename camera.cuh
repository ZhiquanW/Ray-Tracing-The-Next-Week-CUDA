//
// Created by mySab on 2019/06/02.
//

#ifndef RAY_TRACING_ENGINE_CAMERA_CUDA_H
#define RAY_TRACING_ENGINE_CAMERA_CUDA_H
#include "ray.cuh"

__device__ vector3 random_in_unit_disk(curandState *local_rand_state) {
  vector3 p;
  do {
    p = 2.0f * vector3(curand_uniform(local_rand_state),
                       curand_uniform(local_rand_state), 0) -
        vector3(1, 1, 0);
  } while (dot(p, p) >= 1.0f);
  return p;
}
class camera {
public:
  vector3 origin;
  vector3 lower_left_corner;
  vector3 horizontal_vec;
  vector3 vertical_vec;
  vector3 u, v, w;
  float lens_radius;
  __device__ camera(vector3, vector3, vector3, float, float, float, float);
  __device__ ray gen_ray(const float &, const float &, curandState *);
};

__device__
camera::camera(vector3 lookfrom, vector3 lookat, vector3 vup, float vfov,
               float aspect, float aperture,
               float focus_dist) { // vfov is top to bottom in degrees
  lens_radius = aperture / 2.0f;
  float theta = vfov * ((float)M_PI) / 180.0f;
  float half_height = tan(theta / 2.0f);
  float half_width = aspect * half_height;
  origin = lookfrom;
  w = (lookfrom - lookat).normalize();
  u = (cross(vup, w)).normalize();
  v = cross(w, u);
  lower_left_corner = origin - half_width * focus_dist * u -
                      half_height * focus_dist * v - focus_dist * w;
  horizontal_vec = 2.0f * half_width * focus_dist * u;
  vertical_vec = 2.0f * half_height * focus_dist * v;
}

__device__ ray camera::gen_ray(const float &s, const float &t,
                               curandState *local_rand_state) {
  vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
  vector3 offset = u * rd.x() + v * rd.y();
  return ray(origin + offset, lower_left_corner + s * horizontal_vec +
                                  t * vertical_vec - origin - offset);
}

#endif // RAY_TRACING_ENGINE_CAMERA_H