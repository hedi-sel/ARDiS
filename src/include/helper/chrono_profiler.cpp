#include "chrono_profiler.hpp"
#include <assert.h>

chrono_profiler::chrono_profiler() { time = clock(); }

int chrono_profiler::start(std::string name) {
    count_time();

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

void chrono_profiler::end() {
    count_time();
    current = -1;
}

void chrono_profiler::print() {
    count_time();
    for (auto name = names.begin(); name != names.end(); name++) {
        std::cout << name->first << " :\n" << chronos[name->second] << "\n";
    }
}

void chrono_profiler::count_time() {
    if (current >= 0) {
        chronos.at(current) += (double)(clock() - time) / CLOCKS_PER_SEC;
    }
    time = clock();
}