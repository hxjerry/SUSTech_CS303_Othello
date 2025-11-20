#include <cstdint>
#include <utility>

uint64_t player_moves(uint64_t player, uint64_t opponent);
std::pair<uint64_t, uint64_t> flip_pieces(uint64_t player, uint64_t opponent, uint64_t move);
uint64_t zorbist_hash(uint64_t player_board, uint64_t opponent_board);