#include "movement.cuh"
#include <iostream>
using namespace std;

int main() {
  movement tmp_m(0, 0.1, vector3(1.2f, 2.4f, 9.5f));
  cout << tmp_m.start_time() << " - " << tmp_m.time_frame() << " : "
       << tmp_m.velocity() << endl;
  return 0;
}
