#include <cstdint>
#include <immintrin.h>
#include <utility>
#include <xsimd/xsimd.hpp>

#include "include/bitboard.hpp"

namespace xs = xsimd;
/*
Bitboard representation:
  63-56
  ...
  7-0
*/

template<int DIR> uint64_t shift(uint64_t board);
template<> inline uint64_t shift<0>(uint64_t board) { // Up
    return board << 8;
}
template<> inline uint64_t shift<1>(uint64_t board) { // Down
    return board >> 8;
}
template<> inline uint64_t shift<2>(uint64_t board) { // Left
    return (board & 0x7f7f7f7f7f7f7f7f) << 1;
}
template<> inline uint64_t shift<3>(uint64_t board) { // Right
    return (board & 0xfefefefefefefefe) >> 1;
}
template<> inline uint64_t shift<4>(uint64_t board) { // Up-Left
    return (board & 0x7f7f7f7f7f7f7f7f) << 9;
}
template<> inline uint64_t shift<5>(uint64_t board) { // Up-Right
    return (board & 0xfefefefefefefefe) << 7;
}
template<> inline uint64_t shift<6>(uint64_t board) { // Down-Left
    return (board & 0x7f7f7f7f7f7f7f7f) >> 7;
}
template<> inline uint64_t shift<7>(uint64_t board) { // Down-Right
    return (board & 0xfefefefefefefefe) >> 9;
}

const xs::batch<uint64_t, xs::avx2> leftshift_offsets = {8, 1, 9, 7};
const xs::batch<uint64_t, xs::avx2> leftshift_masks = {0xffffffffffffffff, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0xfefefefefefefefe};
const xs::batch<uint64_t, xs::avx2> rightshift_offsets = {8, 1, 7, 9};
const xs::batch<uint64_t, xs::avx2> rightshift_masks = {0xffffffffffffffff, 0xfefefefefefefefe, 0x7f7f7f7f7f7f7f7f, 0xfefefefefefefefe};

uint64_t player_moves(uint64_t player, uint64_t opponent) {

    xs::batch<uint64_t, xs::avx2> candidates_l_vec(player), candidates_r_vec(player);
    xs::batch<uint64_t, xs::avx2> opponent_vec(opponent);
    xs::batch<uint64_t, xs::avx2> empty_vec = ~(candidates_l_vec | opponent_vec);
    xs::batch<uint64_t, xs::avx2> candidates{0};

    candidates_l_vec = ((candidates_l_vec & leftshift_masks) << leftshift_offsets) & opponent_vec;
    candidates_r_vec = ((candidates_r_vec & rightshift_masks) >> rightshift_offsets) & opponent_vec;

    for (int i = 0; i < 6; ++i) {
        auto shifted_l = (candidates_l_vec & leftshift_masks) << leftshift_offsets;
        auto shifted_r = (candidates_r_vec & rightshift_masks) >> rightshift_offsets;
        candidates |= (shifted_l | shifted_r) & empty_vec;
        candidates_l_vec = shifted_l & opponent_vec;
        candidates_r_vec = shifted_r & opponent_vec;
    }

    return xs::reduce([](xs::batch<uint64_t, xs::avx2> a, xs::batch<uint64_t, xs::avx2> b) {
        return a | b;
    }, candidates);
}

std::pair<uint64_t, uint64_t> flip_pieces(uint64_t player, uint64_t opponent, uint64_t move) {
    xs::batch<uint64_t, xs::avx2> move_l_vec(move), move_r_vec(move);
    xs::batch<uint64_t, xs::avx2> l_valid_paths(0xffffffffffffffff), r_valid_paths(0xffffffffffffffff);
    xs::batch<uint64_t, xs::avx2> flip_l_vec(0), flip_r_vec(0);
    xs::batch<uint64_t, xs::avx2> opponent_vec(opponent), not_opponent_vec = ~opponent_vec;

    const xs::batch<uint64_t, xs::avx2> v0 = xs::batch<uint64_t, xs::avx2>(0);
    const xs::batch<uint64_t, xs::avx2> v1 = ~v0;

    // Walk the directions to find connected opponent pieces in each direction
    for (int i = 0; i < 7; ++i) {
        auto shifted_l = (move_l_vec & leftshift_masks) << leftshift_offsets;
        auto shifted_r = (move_r_vec & rightshift_masks) >> rightshift_offsets;

        auto l_hit = shifted_l & opponent_vec;
        auto r_hit = shifted_r & opponent_vec;

        move_l_vec |= l_hit & l_valid_paths;
        move_r_vec |= r_hit & r_valid_paths;
        
        l_valid_paths &= xs::select(xs::neq(shifted_l & not_opponent_vec, v0), v0, v1);
        r_valid_paths &= xs::select(xs::neq(shifted_r & not_opponent_vec, v0), v0, v1);
    }

    // Shift each direction once more, if it hits a player piece, we can flip the pieces in between
    auto final_l = (move_l_vec & leftshift_masks) << leftshift_offsets;
    auto final_r = (move_r_vec & rightshift_masks) >> rightshift_offsets;

    auto player_vec = xs::batch<uint64_t, xs::avx2>(player);
    flip_l_vec = xs::select(xs::neq(final_l & player_vec, v0), move_l_vec, v0);
    flip_r_vec = xs::select(xs::neq(final_r & player_vec, v0), move_r_vec, v0);

    uint64_t flips = xs::reduce([](xs::batch<uint64_t, xs::avx2> a, xs::batch<uint64_t, xs::avx2> b) {
        return a | b;
    }, flip_l_vec | flip_r_vec);
    player |= flips | move;
    opponent &= ~flips;
    return {player, opponent};
}