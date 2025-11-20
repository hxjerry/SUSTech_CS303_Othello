#include <cstdint>
#include <vector>
#include <array>
// #include <unordered_map>
// #include <unordered_set>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <random>
#include "bitboard.hpp"

class mcts_node {
public:

    float visit_count;
    float win_score;

    std::vector<std::tuple<size_t,size_t,float>> children; // (child_player, child_opponent, prior)

    mcts_node() : visit_count(0), win_score(0) {}
    mcts_node(u_int64_t player, u_int64_t opponent, float value, const std::array<float, 64>& policy);
};

class mcts {
private:
    absl::flat_hash_map<std::pair<uint64_t, uint64_t>, mcts_node> transposition_table;
    std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<mcts_node*>>> pending_nodes;
    absl::flat_hash_set<std::pair<uint64_t, uint64_t>> pending_node_set;
    std::mt19937_64 rng;

public:
    mcts() : rng(std::random_device{}()) {}

    std::pair<size_t,size_t> root_key;

    bool select();
    int batch_select(int max_pending, int max_explore);

    std::vector<std::pair<uint64_t, uint64_t>> get_pending_nodes();
    void apply_evaluation(const std::vector<float>& values, const std::vector<std::array<float, 64>>& policies);
    void rebuild_tree(); // Copy tree from root_key
    void take_action(int move_idx);
    bool terminal_state();
    u_int64_t get_best_action(float temperature);
    std::array<float, 64> get_policy();
};