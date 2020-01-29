#include "chrono_profiler.hpp"
#include <assert.h>

ChronoProfiler::ChronoProfiler() { time = clock(); }

int ChronoProfiler::Start(std::string name) {
    CountTime();

    current = names[name] - 1;

    if (current == -1) {
        assert(names.count(name) == 1);
        current = names.size() - 1;
        names[name] = current + 1;
        chronos.push_back(0.0);
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
    for (auto name = names.begin(); name != names.end(); name++) {
        std::cout << name->first << " : " << chronos[name->second] << "s\n";
    }
}

void ChronoProfiler::CountTime() {
    if (current >= 0) {
        chronos.at(current) += (FLOAT)(clock() - time) / CLOCKS_PER_SEC;
    }
    time = clock();
}