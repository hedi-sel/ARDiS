#include "chrono_profiler.hpp"
#include <assert.h>

ChronoProfiler::ChronoProfiler() { time = clock(); }

int ChronoProfiler::Start(std::string name) {
    CountTime();

    auto currentElement = names.find(name);

    if (currentElement == names.end()) {
        current = chronos.size();
        names[name] = current;
        chronos.push_back(0.0);
    } else {
        current = currentElement->second;
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
        std::cout << name->first << " : " << chronos[name->second] << "\n";
    }
}

void ChronoProfiler::CountTime() {
    if (current >= 0) {
        chronos.at(current) += (FLOAT)(clock() - time) / CLOCKS_PER_SEC;
    }
    time = clock();
}