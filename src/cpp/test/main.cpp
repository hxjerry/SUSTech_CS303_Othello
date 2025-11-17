#include <iostream>
#include <chrono>
#include <tuple>
#include <utility>
#include "../include/bitboard.hpp"


void print_board(uint64_t player, uint64_t opponent, uint64_t moves) {
    for (int i = 7; i >= 0; --i) {
        for (int j = 7; j >= 0; --j) {
            uint64_t mask = 1ULL << (i * 8 + j);
            if (player & mask) {
                std::cout << " P ";
            } else if (opponent & mask) {
                std::cout << " O ";
            } else if (moves & mask) {
                std::cout << " * ";
            } else {
                std::cout << " . ";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

int main() {
    return 0;
}