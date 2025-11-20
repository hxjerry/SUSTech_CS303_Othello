#include <array>
#include <cstdint>
#include <random>
#include <xsimd/xsimd.hpp>
#include "include/bitboard.hpp"
namespace xs = xsimd;

std::array<uint64_t, 128> zorbist_table_initialize() {
    std::array<uint64_t, 128> table{};
    std::mt19937_64 rng(42);
    for (auto& entry : table) {
        entry = rng();
    }
    return table;
}

__attribute__((aligned(32))) const auto zorbist_table = zorbist_table_initialize();

uint64_t zorbist_hash(uint64_t player_board, uint64_t opponent_board) {
    xs::batch<uint64_t, xs::avx2> sum_1(0), sum_2(0);
    xs::batch<uint64_t, xs::avx2> tail_mask(1ull);
    xs::batch<uint64_t, xs::avx2> zero(0);

    const xs::batch<uint64_t, xs::avx2> shifts = {0, 16, 32, 48};
    auto board_vec_1 = xs::broadcast<uint64_t, xs::avx2>(player_board) >> shifts;
    auto board_vec_2 = xs::broadcast<uint64_t, xs::avx2>(opponent_board) >> shifts;

    for (int i = 0; i < 16; ++i) {
        auto mask_1 = (board_vec_1 & tail_mask) - 1;
        auto mask_2 = (board_vec_2 & tail_mask) - 1;

        auto table_slice_1 = xs::load_aligned<xs::avx2>(&zorbist_table[i * 8]);
        auto table_slice_2 = xs::load_aligned<xs::avx2>(&zorbist_table[i * 8 + 4]);

        sum_1 ^= mask_1 & table_slice_1;
        sum_2 ^= mask_2 & table_slice_2;

        board_vec_1 >>= 1;
        board_vec_2 >>= 1;
    }

    return xs::reduce([](xs::batch<uint64_t, xs::avx2> a, xs::batch<uint64_t, xs::avx2> b) {
        return a ^ b;
    },sum_1 ^ sum_2);
}
