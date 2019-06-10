#ifndef MATERIAL_CUDA_H
#define MATERIAL_CUDA_H

#include "hitinfo.cuh"
#include "ray.cuh"

__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
  vector3 p;
  do {
    p = 2.0f * vector3(curand_uniform(local_rand_state),
                       curand_uniform(local_rand_state),
                       curand_uniform(local_rand_state)) -
        vector3(1, 1, 1);
  } while (p.squared_length() >= 1.0f);
  return p;
}

__device__ float schlick(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vector3 &v, const vector3 &n, float ni_over_nt,
                        vector3 &refracted) {
  vector3 uv = v.normalize();
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else
    return false;
}

__device__ vector3 reflect(const vector3 &_v, const vector3 &_n) {
  return _v - 2.0f * dot(_v, _n) * _n;
}
class material {
public:
  __device__ virtual bool scatter(const ray &, const hitinfo &, vector3 &,
                                  ray &, curandState *) const = 0;
};

class lambertian : public material {
public:
  vector3 albedo;
  __device__ lambertian(const vector3 &);
  __device__ virtual bool scatter(const ray &, const hitinfo &, vector3 &,
                                  ray &, curandState *) const;
};

__device__ lambertian::lambertian(const vector3 &a) : albedo(a) {}
__device__ bool lambertian::scatter(const ray &r_in, const hitinfo &rec,
                                    vector3 &attenuation, ray &scattered,
                                    curandState *local_rand_state) const {
  vector3 target =
      rec.pos + rec.normal + random_in_unit_sphere(local_rand_state);
  scattered = ray(rec.pos, target - rec.pos);
  attenuation = albedo;
  return true;
}

class metal : public material {
public:
  vector3 albedo;
  float fuzz;
  __device__ metal(const vector3 &, float);
  __device__ virtual bool scatter(const ray &, const hitinfo &, vector3 &,
                                  ray &, curandState *) const;
};
__device__ metal::metal(const vector3 &a, float f) : albedo(a) {
  if (f < 1)
    fuzz = f;
  else
    fuzz = 1;
}
__device__ bool metal::scatter(const ray &r_in, const hitinfo &tmp_info,
                               vector3 &attenuation, ray &scattered,
                               curandState *local_rand_state) const {
  vector3 reflected = reflect(r_in.direction().normalize(), tmp_info.normal);
  scattered = ray(tmp_info.pos,
                  reflected + fuzz * random_in_unit_sphere(local_rand_state));
  attenuation = albedo;
  return (dot(scattered.direction(), tmp_info.normal) > 0.0f);
}

class dielectric : public material {
public:
  float ref_idx; // refractive indice
  __device__ dielectric(float);
  __device__ virtual bool scatter(const ray &, const hitinfo &, vector3 &,
                                  ray &, curandState *) const;
};

__device__ dielectric::dielectric(float ri) : ref_idx(ri) {}
__device__ bool dielectric::scatter(const ray &r_in, const hitinfo &info,
                                    vector3 &attenuation, ray &scattered,
                                    curandState *local_rand_state) const {
  vector3 outward_normal;
  vector3 reflected = reflect(r_in.direction(), info.normal);
  float ni_over_nt;
  attenuation = vector3(1.0, 1.0, 1.0);
  vector3 refracted;
  float reflect_prob;
  float cosine;
  if (dot(r_in.direction(), info.normal) > 0.0f) {
    outward_normal = -1 * info.normal;
    ni_over_nt = ref_idx;
    cosine = dot(r_in.direction(), info.normal) / r_in.direction().length();
    cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
  } else {
    outward_normal = info.normal;
    ni_over_nt = 1.0f / ref_idx;
    cosine = -dot(r_in.direction(), info.normal) / r_in.direction().length();
  }
  if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
    reflect_prob = schlick(cosine, ref_idx);
  else
    reflect_prob = 1.0f;
  if (curand_uniform(local_rand_state) < reflect_prob)
    scattered = ray(info.pos, reflected);
  else
    scattered = ray(info.pos, refracted);
  return true;
}
#endif