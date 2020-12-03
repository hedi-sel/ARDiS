#pragma once

#include <iostream>
#include <map>
#include <string>
#include <time.h>
#include <vector>

#define NCHRONO

class chrono_profiler {
  public:
    chrono_profiler();
    int start(std::string name);
    void end();
    void print();

  private:
    clock_t time;

    std::map<std::string, int> names;
    std::vector<double> chronos;

    int current = -1;

    void count_time();
};