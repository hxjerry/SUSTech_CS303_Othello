#include <iostream>
#include <chrono>
#include <tuple>
#include <utility>
#include "../include/bitboard.hpp"
#include "../include/mcts.hpp"

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
    uint64_t player = 0x0000000810000000;
    uint64_t opponent = 0x0000001008000000;

    mcts tree;
    tree.root_key = {player, opponent};
    
    while (!tree.terminal_state()) {
        for (int i = 0; i < 128; ++i) {
            tree.batch_select(64, 1024);

            auto pending_size = tree.get_pending_nodes().size();

            tree.apply_evaluation(std::vector<float>(pending_size, 0.0f), std::vector<std::array<float, 64>>(pending_size, std::array<float, 64>{}));
        }
        int move_idx = tree.get_best_action(0.0f);
        std::cout << "Selected move index: " << move_idx << std::endl;
        tree.take_action(move_idx);

        print_board(tree.root_key.first, tree.root_key.second, player_moves(tree.root_key.first, tree.root_key.second));
    }

    return 0;
}