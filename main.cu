#include "camera.cuh"
#include "material.cuh"
#include "pngmaster.h"
#include "ray.cuh"
#include "scene.cuh"
#include "shpere.cuh"
#include "vector3.cuh"
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

#define RANDVEC3                                                               \
  vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),  \
          curand_uniform(local_rand_state))

__device__ vector3 color(const ray &_r, scene **_tmp_scene,
                         curandState *local_rand_state) {

  ray cur_ray = _r;
  vector3 cur_attenuation(1.0f, 1.0f, 1.0f);
  for (int i = 0; i < 20; ++i) {
    hitinfo tmp_info;
    if ((*_tmp_scene)->hit(cur_ray, 0.001f, FLT_MAX, tmp_info)) {
      ray scattered_ray;
      vector3 attenutation;
      if (tmp_info.material_ptr->scatter(cur_ray, tmp_info, attenutation,
                                         scattered_ray, local_rand_state)) {
        cur_attenuation *= attenutation;
        cur_ray = scattered_ray;
      } else {
        return vector3(0.0f, 0.0f, 0.0f);
      }
    } else {
      vector3 unit_direction = _r.direction().normalize();
      float t = 0.5f * (unit_direction.y() + 1.0f);
      return cur_attenuation * (1.0f - t) * vector3(1, 1, 1) +
             t * vector3(0.5f, 0.7f, 1.0f);
    }
  }
  return vector3(0.0f, 0.0f, 0.0f);
}

__global__ void render(vector3 *fb, int max_x, int max_y, int ray_num,
                       camera **tmp_cam, scene **tmp_scene,
                       curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];

  vector3 tmp_col(0, 0, 0);
  for (int r = 0; r < ray_num; ++r) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray tmp_r = (*tmp_cam)->gen_ray(u, v, rand_state);
    tmp_col += color(tmp_r, tmp_scene, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;
  tmp_col /= float(ray_num);
  // tmp_col[0] = sqrt(tmp_col[0]);
  // tmp_col[1] = sqrt(tmp_col[1]);
  // tmp_col[2] = sqrt(tmp_col[2]);
  fb[pixel_index] = tmp_col;
}
__global__ void rand_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}
__global__ void init_scene(object **objs, scene **tmp_scene, camera **tmp_cam,
                           int nx, int ny, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    objs[0] = new sphere(vector3(0.0f, -1000.0f, -1.0f), 1000.0f,
                         new lambertian(vector3(0.5f, 0.5f, 0.5f)));
    int i = 1;
    for (int a = -11; a < 11; ++a) {
      for (int b = -11; b < 11; ++b) {
        float choose_mat = (curand_uniform(&local_rand_state));
        vector3 tmp_center(a + curand_uniform(&local_rand_state), 0.2f,
                           b + curand_uniform(&local_rand_state));
        if (choose_mat < 0.8f) {
          objs[i++] = new sphere(
              tmp_center, 0.2f,
              new lambertian(vector3(curand_uniform(&local_rand_state) *
                                         curand_uniform(&local_rand_state),
                                     curand_uniform(&local_rand_state) *
                                         curand_uniform(&local_rand_state),
                                     curand_uniform(&local_rand_state) *
                                         curand_uniform(&local_rand_state))));

        } else if (choose_mat < 0.95f) {
          objs[i++] = new sphere(
              tmp_center, 0.2f,
              new metal(
                  vector3(0.5f * (1.0 + curand_uniform(&local_rand_state)),
                          0.5f * (1.0 + curand_uniform(&local_rand_state)),
                          0.5f * (1.0 + curand_uniform(&local_rand_state))),
                  0.5f * curand_uniform(&local_rand_state)));
        } else {
          objs[i++] = new sphere(tmp_center, 0.2, new dielectric(1.5));
        }
      }
    }
    objs[i++] = new sphere(vector3(0, 1, 0), 1.0, new dielectric(1.5));
    objs[i++] = new sphere(vector3(-4, 1, 0), 1.0,
                           new lambertian(vector3(0.4, 0.2, 0.1)));
    objs[i++] = new sphere(vector3(4, 1, 0), 1.0,
                           new metal(vector3(0.7, 0.6, 0.5), 0.0));
    *rand_state = local_rand_state;
    *(tmp_scene) = new scene(objs, 22 * 22 + 1 + 3);
    vector3 lookfrom(13, 2, 3);
    vector3 lookat(0, 0, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    *tmp_cam = new camera(lookfrom, lookat, vector3(0, 1, 0), 30.0,
                          float(nx) / float(ny), aperture, dist_to_focus);
  }
}
__global__ void free_scene(object **objs, scene **tmp_scene,
                           camera **d_camera) {
  delete *(objs);
  delete *(objs + 1);
  delete *(tmp_scene);
  delete *(d_camera);
}
int main() {
  int nx = 1920;
  int ny = 1080;
  int tx = 8;
  int ty = 8;
  int ray_num = 10;
  const int obj_nums = 22 * 22 + 1 + 3;
  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vector3);
  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  // allocate FB
  vector3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // init scene

  scene **d_tmp_scene;
  checkCudaErrors(cudaMalloc((void **)&d_tmp_scene, sizeof(scene *)));
  object **d_objs;
  checkCudaErrors(cudaMalloc((void **)&d_objs, sizeof(object *) * obj_nums));
  curandState *d_rand_state;
  checkCudaErrors(
      cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
  camera **d_tmp_cam;
  checkCudaErrors(cudaMalloc((void **)&d_tmp_cam, sizeof(camera *)));
  rand_init<<<blocks, threads>>>(nx, ny, d_rand_state);

  init_scene<<<1, 1>>>(d_objs, d_tmp_scene, d_tmp_cam, nx, ny, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(fb, nx, ny, ray_num, d_tmp_cam, d_tmp_scene,
                              d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  float timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << nx << " * " << ny << std::endl;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  pngmaster myImage(ny, nx);
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      vector3 tmp_vec = fb[pixel_index] * 255.99f;
      myImage.set_pixel(i, j, tmp_vec.r(), tmp_vec.g(), tmp_vec.b());
    }
  }

  string file_name = "test" + std::to_string(timer_seconds) + ".png";
  myImage.output(file_name.c_str());
  std::cerr << "render finished" << std::endl;
  // free memory
  checkCudaErrors(cudaDeviceSynchronize());
  free_scene<<<1, 1>>>(d_objs, d_tmp_scene, d_tmp_cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_objs));
  checkCudaErrors(cudaFree(d_tmp_scene));
  checkCudaErrors(cudaFree(fb));
  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}