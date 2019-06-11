// Created by Zhiquan Wang on 2019/06/11
#ifndef MOVEMENT_CUH_CUDA
#define MOVEMENT_CUH_CUDA

#include "vector3"
class movement {
public:
  unsigned int node_num;
  float *time_list;
  vector3 *velocity_list;
  movement(unsigned int _n) : node_num(_n) {}
  float get_time(unsigned int _i) { return time_list[_i]; }
  vector3 get_velocity(unsigned int _i) { return velocity_list[i]; }
}

#endif