#pragma once

#include <iostream>
#include <map>
#include <string>
#include <time.h>
#include <vector>

#define NCHRONO

typedef double FLOAT;

class ChronoProfiler {
  public:
    ChronoProfiler();
    int Start(std::string name);
    void End();
    void Print();

  private:
    clock_t time;

    std::map<std::string, int> nameToInt;
    std::vector<std::string> intToName;
    std::vector<FLOAT> chronos;

    int current = -1;

    void CountTime();
};