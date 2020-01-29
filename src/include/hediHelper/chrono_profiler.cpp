#include "chrono_profiler.hpp"
#include <assert.h>

ChronoProfiler::ChronoProfiler() { time = clock(); }

int ChronoProfiler::Start(std::string name) {
    CountTime();

    current = nameToInt[name] - 1;

    if (current == -1) {
        assert(nameToInt.count(name) == 1);
        current = nameToInt.size() - 1;
        nameToInt[name] = current + 1;
        chronos.push_back(0.0);
        intToName.push_back(name);
    }
    time = clock();
    return current;
}

void ChronoProfiler::End() {
    CountTime();
    current = -1;
}

void ChronoProfiler::Print() {
    CountTime();
    std::cout << "Profiling results:\n";
    for (int i = 0; i < chronos.size(); i++) {
        std::cout << intToName[i] << " : " << chronos[i] << "s\n";
    }
}

void ChronoProfiler::CountTime() {
    if (current >= 0) {
        chronos.at(current) += (FLOAT)(clock() - time) / CLOCKS_PER_SEC;
    }
    time = clock();
}