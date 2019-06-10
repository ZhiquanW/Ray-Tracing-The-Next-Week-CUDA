//
// Created by ZhiquanWang on 2018/7/9.
//
#ifndef _SH_PNGMASTER_
#define _SH_PNGMASTER_

#include "svpng.inc"

#include <cstring>
#include <iostream>

using namespace std;

class pngmaster {
private:
  unsigned int height;          //图片像素高度
  unsigned int width;           //图片像素宽度
  unsigned char *data;          //存储颜色信息指针
  const unsigned int dimension; //颜色维度

public:
  pngmaster(unsigned int _h, unsigned int _w) : dimension(4) {
    height = _h;
    width = _w;
    data = new unsigned char[_h * _w * dimension];
    memset(data, 0, _h * _w * dimension);
  }

  void set_pixel(int _x, int _y, int _r, int _g, int _b, int _a = 255) {
    _y = height - _y - 1;
    data[(_y * width + _x) * 4] = (unsigned char)std::min(_r, 255);
    data[(_y * width + _x) * 4 + 1] = (unsigned char)std::min(_g, 255);
    data[(_y * width + _x) * 4 + 2] = (unsigned char)std::min(_b, 255);
    data[(_y * width + _x) * 4 + 3] = (unsigned char)std::min(_a, 255);
  }

  void output(const char *_name) {
    FILE *fp = fopen(_name, "wb");
    svpng(fp, width, height, data, 1);
    fclose(fp);
  }
};

#endif
