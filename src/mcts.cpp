#include <xsimd/xsimd.hpp>
#include <cmath>
#include "include/mcts.hpp"
#include "include/bitboard.hpp"
#include <exception>
#include <limits>
#include <iostream>

namespace xs = xsimd;

inline float score(uint64_t player_board, uint64_t opponent_board) {
    int player_count = __builtin_popcountll(player_board);
    int opponent_count = __builtin_popcountll(opponent_board);
    if (player_count > opponent_count)
        return -1.0f;
    else if (player_count < opponent_count)
        return 1.0f;
    else
        return 0.0f;
}

// TODO: Use simd in softmax
mcts_node::mcts_node(u_int64_t player, u_int64_t opponent, float value, const std::array<float, 64>& policy) {
    visit_count = 1.0f;
    
    size_t possible_moves = player_moves(player, opponent);
    if (possible_moves == 0) {
        if (player_moves(opponent, player) == 0) { // Terminal state
            win_score = score(player, opponent);
            return;
        }
        children.emplace_back(opponent, player, 1.0f); // Pass move
    } else {
        __attribute__((aligned(32))) std::array<float, 64> policy_filtered;
        size_t child_count = 0;
        float policy_max = -1e9;
        for (size_t k = possible_moves, child_count=0; k; ++child_count) {
            size_t move = k & -k;
            policy_filtered[child_count] = policy[__builtin_ctzll(move)];
            if (policy_filtered[child_count] > policy_max) {
                policy_max = policy_filtered[child_count];
            }
            k -= move;
        }
        children.reserve(child_count);
        
        auto t = xs::broadcast<float, xs::avx2>(policy_max);
        for (size_t i = 0; i < child_count; i += 8) {
            auto p_vec = xs::load_aligned<xs::avx2>(&policy_filtered[i]);
            p_vec = xs::exp(p_vec - t);
            xs::store_aligned(&policy_filtered[i], p_vec);
        }
        float policy_sum = 0.0f;
        for (size_t i = 0; i < child_count; ++i) {
            policy_sum += policy_filtered[i];
        }
        t = xs::broadcast<float, xs::avx2>(policy_sum);
        for (size_t i = 0; i < child_count; i += 8) {
            auto p_vec = xs::load_aligned<xs::avx2>(&policy_filtered[i]);
            p_vec = p_vec / t;
            xs::store_aligned(&policy_filtered[i], p_vec);
        }

        for (size_t k = possible_moves, child_idx = 0; k; ++child_idx) {
            size_t move = k & -k;
            auto [child_player, child_opponent] = flip_pieces(player, opponent, move);
            children.emplace_back(child_opponent, child_player, policy_filtered[child_idx]);
            k -= move;
        }
    }
    win_score = value;
}

bool mcts::select() {
    auto [player,opponent] = root_key;
    int path_length = 1;
    std::vector<mcts_node*> nodes_in_path;

    while (true) {
        auto it = transposition_table.find({player, opponent});

        if (it == transposition_table.end())
            break;
        mcts_node& node = it->second;
        nodes_in_path.push_back(&node);
        if (node.children.empty())
            break;

        float best_puct = -std::numeric_limits<float>::infinity();
        int best_child_idx = -1;
        float sqrt_visit = std::sqrt(node.visit_count);

        std::array<float, 64> puct_values;

        for (size_t i = 0; i < node.children.size(); ++i) {
            auto [child_player, child_opponent, prior] = node.children[i];
            
            auto child_it = transposition_table.find({child_player, child_opponent});
            float q_value = 0.0f;
            float child_visit_count = 0.0f;
            if (child_it != transposition_table.end()) {
                mcts_node& child_node = child_it->second;
                q_value = -child_node.win_score / (child_node.visit_count + 1e-5f);
                child_visit_count = child_node.visit_count;
            } else if (pending_node_set.find({child_player, child_opponent}) != pending_node_set.end()) { // Penalize pending nodes
                q_value = -1.0f;
            }

            puct_values[i] = q_value + 1.5f * prior * (sqrt_visit / (1.0f + child_visit_count));
        }

        for (size_t i = 0; i < node.children.size(); ++i) {
            if (puct_values[i] > best_puct) {
                best_puct = puct_values[i];
                best_child_idx = i;
            }
        }

        auto [next_player, next_opponent, _] = node.children[best_child_idx];
        player = next_player;
        opponent = next_opponent;
        path_length ++;
    }

    if (nodes_in_path.size() == path_length) { // Leaf node already exists
        float leaf_score = score(player, opponent);
        for (int i = nodes_in_path.size() - 1; i >= 0; --i) {
            nodes_in_path[i]->visit_count += 1.0f;
            nodes_in_path[i]->win_score += leaf_score;
            leaf_score = -leaf_score;
        }
        return true;
    }

    if (pending_node_set.find({player, opponent}) != pending_node_set.end()) {
        return false; // Already pending evaluation
    }

    // Assign visit count and penalty to nodes leading to pending evaluation
    for(auto node_ptr : nodes_in_path){
        node_ptr->visit_count += 1.0f;
        node_ptr->win_score += 1.0f; // Penalize against selecting this path
    }

    // Transfer leaf node to pending nodes for evaluation
    pending_node_set.emplace(player, opponent);
    pending_nodes.emplace_back(std::make_pair(player, opponent), nodes_in_path);
    return true;
}

int mcts::batch_select(int max_pending, int max_explore) {
    for (int i = 0; i < max_explore; ++i) {
        if (!select() || pending_nodes.size() >= max_pending) {
            break;
        }
    }
    return pending_nodes.size();
}

std::vector<std::pair<uint64_t, uint64_t>> mcts::get_pending_nodes() {
    std::vector<std::pair<uint64_t, uint64_t>> nodes;
    nodes.reserve(pending_nodes.size());
    for (const auto& node : pending_nodes) {
        nodes.emplace_back(node.first);
    }
    return nodes;
}

void mcts::apply_evaluation(const std::vector<float>& values, const std::vector<std::array<float, 64>>& policies) {
    if (values.size() != pending_nodes.size() || policies.size() != pending_nodes.size()) {
        throw std::runtime_error("Mismatched evaluation sizes.");
    }

    // modifty path in one go before changing the transposition table
    for (size_t i = 0; i < pending_nodes.size(); ++i) {
        float score = values[i];
        auto path = pending_nodes[i].second;
        for (int j = path.size() - 1; j >= 0; --j) {
            score = -score;
            path[j]->win_score += score - 1.0f;
        }
    }
    // Construct new leaf nodes
    for (size_t i = 0; i < pending_nodes.size(); ++i) {
        auto [player, opponent] = pending_nodes[i].first;
        mcts_node new_node(player, opponent, values[i], policies[i]);
        transposition_table.emplace(std::make_pair(player, opponent), std::move(new_node));
        pending_node_set.erase({player, opponent});
    }
    
    pending_nodes.clear();
    pending_node_set.clear();
    return;
}

void dfs_rebuild(std::pair<uint64_t, uint64_t> root, absl::flat_hash_map<std::pair<uint64_t, uint64_t>, mcts_node>& old_table, absl::flat_hash_map<std::pair<uint64_t, uint64_t>, mcts_node>& new_table) {
    if (new_table.find(root) != new_table.end())
        return;
    auto node = old_table.extract(root);
    if (node.empty())
        return;
    const auto new_node = new_table.insert(std::move(node)).position->second;
    const auto &children = new_node.children;

    for (const auto& [child_player, child_opponent, _] : children) {
        dfs_rebuild({child_player, child_opponent}, old_table, new_table);
    }
    return;
}

void mcts::rebuild_tree() {
    absl::flat_hash_map<std::pair<uint64_t, uint64_t>, mcts_node> new_table;
    new_table.reserve(transposition_table.size()*2);
    dfs_rebuild(root_key, transposition_table, new_table);
    transposition_table = std::move(new_table);
}

void mcts::take_action(int move_idx) {
    if (terminal_state()) {
        throw std::runtime_error("Cannot take action in terminal state.");
    }
    uint64_t move = 1ULL << move_idx;
    uint64_t possible_moves = player_moves(root_key.first, root_key.second);
    if (possible_moves == 0) { // Pass move
        root_key = {root_key.second, root_key.first}; // Switch perspectives
    } else {
        if ((possible_moves & move) == 0) {
            throw std::runtime_error("Invalid move taken.");
        }
        auto [new_player, new_opponent] = flip_pieces(root_key.first, root_key.second, move);
        root_key = {new_opponent, new_player}; // Switch perspectives
    }
    rebuild_tree();
    return;
}

bool mcts::terminal_state() {
    uint64_t player_moves_available = player_moves(root_key.first, root_key.second);
    uint64_t opponent_moves_available = player_moves(root_key.second, root_key.first);
    return (player_moves_available == 0 && opponent_moves_available == 0);
}

u_int64_t mcts::get_best_action(float temperature) {
    auto it = transposition_table.find(root_key);
    if (it == transposition_table.end()) {
        throw std::runtime_error("Root node not found in transposition table.");
    }
    mcts_node& root_node = it->second;

    __attribute__((aligned(32))) float visit_counts[64];
    for (size_t i = 0; i < root_node.children.size(); ++i) {
        auto [child_player, child_opponent, _] = root_node.children[i];
        auto child_it = transposition_table.find({child_player, child_opponent});
        if (child_it != transposition_table.end()) {
            mcts_node& child_node = child_it->second;
            visit_counts[i] = child_node.visit_count;
        } else {
            visit_counts[i] = 0.0f;
        }
    }

    size_t best_move = 0;
    if (temperature != 0) {
        float max_value = -1e9;
        for (float v : visit_counts)
            if (v > max_value)
                max_value = v;
        auto t_max = xs::broadcast<float,xs::avx2>(max_value);
        auto t_temp = xs::broadcast<float,xs::avx2>(temperature);
        for (size_t i = 0; i < root_node.children.size(); i += 8) {
            auto v_vec = xs::load_aligned<xs::avx2>(&visit_counts[i]);
            v_vec = xs::exp((v_vec - t_max) / t_temp);
            xs::store_aligned(&visit_counts[i], v_vec);
        }
        std::discrete_distribution<size_t> dist(visit_counts, visit_counts + root_node.children.size());
        best_move = dist(rng);
    } else {
        float max_value = -1e9;
        for (size_t i = 0; i < root_node.children.size(); ++i)
            if (visit_counts[i] > max_value) {
                max_value = visit_counts[i];
                best_move = i;
            }
    }
    auto [child_player, child_opponent, _] = root_node.children[best_move];
    auto [player, opponent] = root_key;
    return __builtin_ctzll((child_player | child_opponent) - (player | opponent));
}

std::array<float, 64> mcts::get_policy() {
    auto it = transposition_table.find(root_key);
    if (it == transposition_table.end()) {
        throw std::runtime_error("Root node not found in transposition table.");
    }
    mcts_node& root_node = it->second;

    __attribute__((aligned(32))) float visit_counts[64] = {0.0f};
    float visit_sum = 0.0f;
    for (size_t i = 0; i < root_node.children.size(); ++i) {
        auto [child_player, child_opponent, _] = root_node.children[i];
        auto child_it = transposition_table.find({child_player, child_opponent});
        int child_idx = __builtin_ctzll((child_player | child_opponent) - (root_key.first | root_key.second));
        
        if (child_it != transposition_table.end()) {
            mcts_node& child_node = child_it->second;
            visit_counts[child_idx] = child_node.visit_count;
            visit_sum += child_node.visit_count;
        }
    }
    std::array<float, 64> policy{};
    if (visit_sum > 0.0f)
        for (size_t i = 0; i < 64; ++i) {
            policy[i] = visit_counts[i] / visit_sum;
        }
    return policy;
}