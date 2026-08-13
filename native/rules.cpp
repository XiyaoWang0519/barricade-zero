#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
uint64_t rotate_anchor_mask(uint64_t mask, int width) {
    uint64_t rotated = 0;
    int anchors = width * width;
    for (int index = 0; index < anchors; ++index)
        if (mask & (uint64_t{1} << index))
            rotated |= uint64_t{1} << (anchors - 1 - index);
    return rotated;
}

struct Board {
    using CellBits = unsigned __int128;

    int n, p[2], walls[2], turn, winner;
    uint64_t h, v;

    bool anchor(uint64_t mask, int r, int c) const {
        if (r < 0 || c < 0 || r >= n - 1 || c >= n - 1) return false;
        return mask & (uint64_t{1} << (r * (n - 1) + c));
    }
    bool blocked(int a, int b, uint64_t hm, uint64_t vm) const {
        int ar = a / n, ac = a % n, br = b / n, bc = b % n;
        if (ar != br) {
            int upper = std::min(ar, br);
            return anchor(hm, upper, ac) || anchor(hm, upper, ac - 1);
        }
        int left = std::min(ac, bc);
        return anchor(vm, ar, left) || anchor(vm, ar - 1, left);
    }
    bool path(int player, int extra, int er, int ec) const {
        uint64_t hm = h, vm = v;
        if (extra == 1) hm |= uint64_t{1} << (er * (n - 1) + ec);
        if (extra == 2) vm |= uint64_t{1} << (er * (n - 1) + ec);
        int goal = player == 0 ? 0 : n - 1;
        int queue[81];
        uint8_t seen[81] = {};
        int front = 0, back = 0;
        queue[back++] = p[player];
        seen[p[player]] = 1;
        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};
        while (front < back) {
            int cell = queue[front++];
            int r = cell / n, c = cell % n;
            if (r == goal) return true;
            for (int i = 0; i < 4; ++i) {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
                int next = nr * n + nc;
                if (!seen[next] && !blocked(cell, next, hm, vm)) {
                    seen[next] = 1;
                    queue[back++] = next;
                }
            }
        }
        return false;
    }
    bool geometry(bool horizontal, int r, int c) const {
        if (anchor(h, r, c) || anchor(v, r, c)) return false;
        if (horizontal)
            return !anchor(h, r, c - 1) && !anchor(h, r, c + 1);
        return !anchor(v, r - 1, c) && !anchor(v, r + 1, c);
    }
    void build_open_edges(uint8_t* open) const {
        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};
        int cells = n * n;
        std::fill(open, open + cells, uint8_t{0});
        for (int cell = 0; cell < cells; ++cell) {
            int row = cell / n, col = cell % n;
            for (int direction = 0; direction < 4; ++direction) {
                int nr = row + dr[direction], nc = col + dc[direction];
                if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
                if (!blocked(cell, nr * n + nc, h, v))
                    open[cell] |= uint8_t{1} << direction;
            }
        }
    }
    void clear_edge(uint8_t* open, int first, int second) const {
        int delta = second - first;
        int forward = delta == -n ? 0 : delta == n ? 1 : delta == -1 ? 2 : 3;
        int reverse = forward ^ 1;
        open[first] &= ~(uint8_t{1} << forward);
        open[second] &= ~(uint8_t{1} << reverse);
    }
    bool path_on_open_edges(int player, const uint8_t* open) const {
        int goal = player == 0 ? 0 : n - 1;
        int queue[81];
        uint8_t seen[81] = {};
        int front = 0, back = 0;
        queue[back++] = p[player];
        seen[p[player]] = 1;
        constexpr int delta[4] = {-9, 9, -1, 1};
        while (front < back) {
            int cell = queue[front++];
            if (cell / n == goal) return true;
            for (int direction = 0; direction < 4; ++direction) {
                if (!(open[cell] & (uint8_t{1} << direction))) continue;
                int step = direction == 0 ? -n : direction == 1 ? n : delta[direction];
                int next = cell + step;
                if (!seen[next]) {
                    seen[next] = 1;
                    queue[back++] = next;
                }
            }
        }
        return false;
    }
    void build_open_moves(CellBits* open) const {
        std::fill(open, open + 4, CellBits{0});
        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};
        int cells = n * n;
        for (int cell = 0; cell < cells; ++cell) {
            int row = cell / n, col = cell % n;
            CellBits bit = CellBits{1} << cell;
            for (int direction = 0; direction < 4; ++direction) {
                int nr = row + dr[direction], nc = col + dc[direction];
                if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
                if (!blocked(cell, nr * n + nc, h, v)) open[direction] |= bit;
            }
        }
    }
    void block_candidate_wall(CellBits* open, bool horizontal, int row, int col) const {
        if (horizontal) {
            for (int offset = 0; offset < 2; ++offset) {
                int upper = row * n + col + offset;
                int lower = (row + 1) * n + col + offset;
                open[1] &= ~(CellBits{1} << upper);
                open[0] &= ~(CellBits{1} << lower);
            }
        } else {
            for (int offset = 0; offset < 2; ++offset) {
                int left = (row + offset) * n + col;
                int right = left + 1;
                open[3] &= ~(CellBits{1} << left);
                open[2] &= ~(CellBits{1} << right);
            }
        }
    }
    bool path_on_open_moves(int player, const CellBits* open) const {
        CellBits seen = CellBits{1} << p[player];
        CellBits frontier = seen;
        CellBits goal = ((CellBits{1} << n) - 1)
            << ((player == 0 ? 0 : n - 1) * n);
        while (frontier) {
            if (frontier & goal) return true;
            CellBits next = ((frontier & open[0]) >> n)
                | ((frontier & open[1]) << n)
                | ((frontier & open[2]) >> 1)
                | ((frontier & open[3]) << 1);
            next &= ~seen;
            seen |= next;
            frontier = next;
        }
        return false;
    }
    void pawn_actions(std::vector<int>& out) const {
        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};
        constexpr int action[4] = {0, 1, 2, 3};
        int me = p[turn], opp = p[1 - turn], r = me / n, c = me % n;
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
            int adjacent = nr * n + nc;
            if (blocked(me, adjacent, h, v)) continue;
            if (adjacent != opp) { out.push_back(action[i]); continue; }
            int br = nr + dr[i], bc = nc + dc[i];
            if (br >= 0 && bc >= 0 && br < n && bc < n && !blocked(opp, br*n+bc, h, v)) {
                out.push_back(action[i]); continue;
            }
            int side_dr[2] = {dc[i], -dc[i]};
            int side_dc[2] = {dr[i], -dr[i]};
            for (int j = 0; j < 2; ++j) {
                int xr = nr + side_dr[j], xc = nc + side_dc[j];
                if (xr < 0 || xc < 0 || xr >= n || xc >= n || blocked(opp, xr*n+xc, h, v)) continue;
                int rr = dr[i] + side_dr[j], cc = dc[i] + side_dc[j];
                int diagonal = rr == -1 ? (cc == -1 ? 4 : 5) : (cc == -1 ? 6 : 7);
                out.push_back(diagonal);
            }
        }
        std::sort(out.begin(), out.end());
    }
    std::vector<int> legal_actions() const {
        std::vector<int> actions;
        if (winner >= 0) return actions;
        pawn_actions(actions);
        if (walls[turn] > 0) {
            int width = n - 1;
            CellBits base_open[4];
            CellBits candidate_open[4];
            build_open_moves(base_open);
            for (int orientation = 0; orientation < 2; ++orientation)
                for (int r = 0; r < width; ++r)
                    for (int c = 0; c < width; ++c) {
                        if (!geometry(orientation == 0, r, c)) continue;
                        std::copy(base_open, base_open + 4, candidate_open);
                        block_candidate_wall(candidate_open, orientation == 0, r, c);
                        if (path_on_open_moves(0, candidate_open)
                            && path_on_open_moves(1, candidate_open))
                            actions.push_back(8 + orientation * width * width + r * width + c);
                    }
        }
        return actions;
    }
    Board apply_known_action(int action) const {
        Board next = *this;
        next.winner = -1;
        if (action < 8) {
            constexpr int dr[4] = {-1, 1, 0, 0};
            constexpr int dc[4] = {0, 0, -1, 1};
            constexpr int cardinal_action[4] = {0, 1, 2, 3};
            int me = p[turn], opponent = p[1 - turn];
            int row = me / n, col = me % n;
            int destination = -1;
            for (int direction = 0; direction < 4 && destination < 0; ++direction) {
                int nr = row + dr[direction], nc = col + dc[direction];
                if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
                int adjacent = nr * n + nc;
                if (blocked(me, adjacent, h, v)) continue;
                if (adjacent != opponent) {
                    if (action == cardinal_action[direction]) destination = adjacent;
                    continue;
                }
                int br = nr + dr[direction], bc = nc + dc[direction];
                if (br >= 0 && bc >= 0 && br < n && bc < n
                    && !blocked(opponent, br * n + bc, h, v)) {
                    if (action == cardinal_action[direction]) destination = br * n + bc;
                    continue;
                }
                int side_dr[2] = {dc[direction], -dc[direction]};
                int side_dc[2] = {dr[direction], -dr[direction]};
                for (int side = 0; side < 2; ++side) {
                    int xr = nr + side_dr[side], xc = nc + side_dc[side];
                    if (xr < 0 || xc < 0 || xr >= n || xc >= n
                        || blocked(opponent, xr * n + xc, h, v)) continue;
                    int rr = dr[direction] + side_dr[side];
                    int cc = dc[direction] + side_dc[side];
                    int diagonal = rr == -1 ? (cc == -1 ? 4 : 5) : (cc == -1 ? 6 : 7);
                    if (action == diagonal) destination = xr * n + xc;
                }
            }
            if (destination < 0) {
                assert(destination >= 0 && "known-legal pawn action was not applicable");
                next.winner = -2;
                return next;
            }
            next.p[turn] = destination;
            int goal = turn == 0 ? 0 : n - 1;
            if (destination / n == goal) next.winner = turn;
        } else {
            int width = n - 1;
            int index = action - 8;
            int orientation = index / (width * width);
            index %= width * width;
            if (orientation == 0) next.h |= uint64_t{1} << index;
            else next.v |= uint64_t{1} << index;
            --next.walls[turn];
        }
        next.turn = 1 - turn;
        return next;
    }
    template <typename T>
    void distance_plane_on_open(
        int player, T* output, const CellBits* open
    ) const {
        int cells = n * n;
        std::fill(output, output + cells, T{0});
        int goal = player == 0 ? 0 : n - 1;
        CellBits frontier = ((CellBits{1} << n) - 1) << (goal * n);
        CellBits seen = frontier;
        T scale = static_cast<T>(std::max(1, cells - 1));
        int distance = 0;
        while (frontier) {
            for (int cell = 0; cell < cells; ++cell) {
                if (frontier & (CellBits{1} << cell)) {
                    output[cell] = static_cast<T>(distance) / scale;
                }
            }
            CellBits next = ((frontier & open[0]) >> n)
                | ((frontier & open[1]) << n)
                | ((frontier & open[2]) >> 1)
                | ((frontier & open[3]) << 1);
            next &= ~seen;
            seen |= next;
            frontier = next;
            ++distance;
        }
    }
    template <typename T>
    void distance_planes(T* first, T* second) const {
        CellBits open[4];
        build_open_moves(open);
        distance_plane_on_open(0, first, open);
        distance_plane_on_open(1, second, open);
    }
};

int rotate_action_180(int action, int n) {
    constexpr int rotated_move[8] = {1, 0, 3, 2, 7, 6, 5, 4};
    if (action < 8) return rotated_move[action];
    int width = n - 1;
    int index = action - 8;
    int orientation = index / (width * width);
    index %= width * width;
    int row = index / width, col = index % width;
    return 8 + orientation * width * width
        + (width - 1 - row) * width + (width - 1 - col);
}

struct NativeEdge {
    int action;
    double prior;
    int visits = 0;
    double value_sum = 0.0;
    int child = -1;
};

struct NativeNode {
    Board board;
    std::vector<NativeEdge> edges;
    bool expanded = false;
    int total_visits = 0;
};

struct NativePending {
    int leaf;
    std::vector<std::pair<int, int>> path;
    bool terminal;
    double terminal_value;
    int evaluation_index = -1;
};

struct NativeSearch {
    int n;
    int action_size;
    double c_puct;
    std::vector<NativeNode> nodes;
    std::vector<NativePending> pending;
    long long expansions = 0;
    long long descents = 0;
    double traversal_seconds = 0.0;
    double preparation_seconds = 0.0;
    double encoding_seconds = 0.0;
    double legality_seconds = 0.0;
    double expansion_seconds = 0.0;

    int add_node(const Board& board) {
        nodes.push_back(NativeNode{board, {}, false, 0});
        return static_cast<int>(nodes.size()) - 1;
    }

    int select_edge(int node_id) const {
        const NativeNode& node = nodes[node_id];
        double scale = std::sqrt(node.total_visits + 1.0);
        int best = 0;
        double best_score = -INFINITY;
        for (int index = 0; index < static_cast<int>(node.edges.size()); ++index) {
            const NativeEdge& edge = node.edges[index];
            double q = edge.visits ? edge.value_sum / edge.visits : 0.0;
            double score = q + c_puct * edge.prior * scale / (1 + edge.visits);
            if (score > best_score) {
                best_score = score;
                best = index;
            }
        }
        return best;
    }

    void set_root_pending(const int* roots, int count) {
        pending.clear();
        pending.reserve(count);
        for (int index = 0; index < count; ++index) {
            if (nodes[roots[index]].expanded) continue;
            const Board& board = nodes[roots[index]].board;
            bool terminal = board.winner >= 0;
            double value = terminal && board.winner == board.turn ? 1.0 : -1.0;
            pending.push_back(NativePending{roots[index], {}, terminal, value});
        }
    }

    void descend(const int* roots, int count) {
        auto started = std::chrono::steady_clock::now();
        pending.clear();
        pending.reserve(count);
        for (int root_index = 0; root_index < count; ++root_index) {
            int node_id = roots[root_index];
            std::vector<std::pair<int, int>> path;
            while (nodes[node_id].expanded && nodes[node_id].board.winner < 0) {
                int edge_index = select_edge(node_id);
                int child = nodes[node_id].edges[edge_index].child;
                path.emplace_back(node_id, edge_index);
                if (child < 0) {
                    Board child_board = nodes[node_id].board.apply_known_action(
                        nodes[node_id].edges[edge_index].action
                    );
                    child = add_node(child_board);
                    nodes[node_id].edges[edge_index].child = child;
                }
                node_id = child;
            }
            const Board& board = nodes[node_id].board;
            bool terminal = board.winner >= 0;
            double value = terminal && board.winner == board.turn ? 1.0 : -1.0;
            pending.push_back(NativePending{
                node_id, std::move(path), terminal, value
            });
        }
        descents += count;
        traversal_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
    }

    void backup(const NativePending& item, double leaf_value) {
        double value = leaf_value;
        for (auto iterator = item.path.rbegin(); iterator != item.path.rend(); ++iterator) {
            value = -value;
            NativeNode& node = nodes[iterator->first];
            NativeEdge& edge = node.edges[iterator->second];
            ++edge.visits;
            edge.value_sum += value;
            ++node.total_visits;
        }
    }
};
}

extern "C" int bz_has_path(int n, int p0, int p1, uint64_t h, uint64_t v,
                            int player, int extra, int row, int col) {
    Board b{n, {p0,p1}, {0,0}, 0, -1, h, v};
    return b.path(player, extra, row, col) ? 1 : 0;
}

extern "C" int bz_legal_actions(int n, int p0, int p1, uint64_t h, uint64_t v,
                                 int w0, int w1, int turn, int winner,
                                 int* output, int capacity) {
    Board b{n, {p0,p1}, {w0,w1}, turn, winner, h, v};
    std::vector<int> actions = b.legal_actions();
    if (static_cast<int>(actions.size()) > capacity) return -static_cast<int>(actions.size());
    std::copy(actions.begin(), actions.end(), output);
    return static_cast<int>(actions.size());
}

extern "C" int bz_legal_actions_batch(
    int n, int count, const int* p0, const int* p1,
    const uint64_t* h, const uint64_t* v, const int* w0, const int* w1,
    const int* turn, const int* winner, int* offsets, int* output, int capacity) {
    int cursor = 0;
    offsets[0] = 0;
    for (int index = 0; index < count; ++index) {
        Board board{n, {p0[index], p1[index]}, {w0[index], w1[index]},
                    turn[index], winner[index], h[index], v[index]};
        std::vector<int> actions = board.legal_actions();
        if (cursor + static_cast<int>(actions.size()) > capacity)
            return -(cursor + static_cast<int>(actions.size()));
        std::copy(actions.begin(), actions.end(), output + cursor);
        cursor += static_cast<int>(actions.size());
        offsets[index + 1] = cursor;
    }
    return cursor;
}

extern "C" int bz_encode_state(int n, int p0, int p1, uint64_t h, uint64_t v,
                                int w0, int w1, double* output, int capacity) {
    int cells = n * n;
    if (capacity < 8 * cells) return -8 * cells;
    std::fill(output, output + 8 * cells, 0.0);
    output[p0] = 1.0;
    output[cells + p1] = 1.0;
    int width = n - 1;
    for (int row = 0; row < width; ++row)
        for (int col = 0; col < width; ++col) {
            uint64_t bit = uint64_t{1} << (row * width + col);
            if (h & bit) output[2 * cells + row * n + col] = 1.0;
            if (v & bit) output[3 * cells + row * n + col] = 1.0;
        }
    double max_walls = std::max(1, std::max(w0, w1));
    std::fill(output + 4 * cells, output + 5 * cells, w0 / max_walls);
    std::fill(output + 5 * cells, output + 6 * cells, w1 / max_walls);
    Board board{n, {p0,p1}, {w0,w1}, 0, -1, h, v};
    board.distance_planes(output + 6 * cells, output + 7 * cells);
    return 8 * cells;
}

extern "C" int bz_encode_state_f32(int n, int p0, int p1, uint64_t h, uint64_t v,
                                    int w0, int w1, float* output, int capacity) {
    int cells = n * n;
    if (capacity < 8 * cells) return -8 * cells;
    std::fill(output, output + 8 * cells, 0.0f);
    output[p0] = 1.0f;
    output[cells + p1] = 1.0f;
    int width = n - 1;
    for (int row = 0; row < width; ++row)
        for (int col = 0; col < width; ++col) {
            uint64_t bit = uint64_t{1} << (row * width + col);
            if (h & bit) output[2 * cells + row * n + col] = 1.0f;
            if (v & bit) output[3 * cells + row * n + col] = 1.0f;
        }
    float max_walls = static_cast<float>(std::max(1, std::max(w0, w1)));
    std::fill(output + 4 * cells, output + 5 * cells, w0 / max_walls);
    std::fill(output + 5 * cells, output + 6 * cells, w1 / max_walls);
    Board board{n, {p0,p1}, {w0,w1}, 0, -1, h, v};
    board.distance_planes(output + 6 * cells, output + 7 * cells);
    return 8 * cells;
}

extern "C" int bz_encode_state_batch_f32(
    int n, int count, const int* p0, const int* p1,
    const uint64_t* h, const uint64_t* v, const int* w0, const int* w1,
    float* output, int capacity) {
    int per_state = 8 * n * n;
    int required = count * per_state;
    if (capacity < required) return -required;
    for (int index = 0; index < count; ++index) {
        int result = bz_encode_state_f32(
            n, p0[index], p1[index], h[index], v[index], w0[index], w1[index],
            output + index * per_state, per_state
        );
        if (result != per_state) return -required;
    }
    return required;
}

extern "C" int bz_encode_canonical_batch_f32(
    int n, int count, const int* p0, const int* p1,
    const uint64_t* h, const uint64_t* v, const int* w0, const int* w1,
    const int* turn, float* output, int capacity) {
    int per_state = 8 * n * n;
    int required = count * per_state;
    if (capacity < required) return -required;
    int cells = n * n;
    for (int index = 0; index < count; ++index) {
        int canonical_p0 = p0[index];
        int canonical_p1 = p1[index];
        uint64_t canonical_h = h[index];
        uint64_t canonical_v = v[index];
        int canonical_w0 = w0[index];
        int canonical_w1 = w1[index];
        if (turn[index] == 1) {
            canonical_p0 = cells - 1 - p1[index];
            canonical_p1 = cells - 1 - p0[index];
            canonical_h = rotate_anchor_mask(h[index], n - 1);
            canonical_v = rotate_anchor_mask(v[index], n - 1);
            canonical_w0 = w1[index];
            canonical_w1 = w0[index];
        }
        int result = bz_encode_state_f32(
            n, canonical_p0, canonical_p1, canonical_h, canonical_v,
            canonical_w0, canonical_w1, output + index * per_state, per_state
        );
        if (result != per_state) return -required;
    }
    return required;
}

extern "C" int bz_prepare_batch_f32(
    int n, int count, const int* p0, const int* p1,
    const uint64_t* h, const uint64_t* v, const int* w0, const int* w1,
    const int* turn, const int* winner, float* encoded, int encoded_capacity,
    int* offsets, int* actions, int action_capacity) {
    int per_state = 8 * n * n;
    int required = count * per_state;
    if (encoded_capacity < required) return -required;
    int cells = n * n;
    int cursor = 0;
    offsets[0] = 0;
    for (int index = 0; index < count; ++index) {
        int canonical_p0 = p0[index];
        int canonical_p1 = p1[index];
        uint64_t canonical_h = h[index];
        uint64_t canonical_v = v[index];
        int canonical_w0 = w0[index];
        int canonical_w1 = w1[index];
        if (turn[index] == 1) {
            canonical_p0 = cells - 1 - p1[index];
            canonical_p1 = cells - 1 - p0[index];
            canonical_h = rotate_anchor_mask(h[index], n - 1);
            canonical_v = rotate_anchor_mask(v[index], n - 1);
            canonical_w0 = w1[index];
            canonical_w1 = w0[index];
        }
        int encoded_count = bz_encode_state_f32(
            n, canonical_p0, canonical_p1, canonical_h, canonical_v,
            canonical_w0, canonical_w1, encoded + index * per_state, per_state
        );
        if (encoded_count != per_state) return -required;

        Board board{n, {p0[index], p1[index]}, {w0[index], w1[index]},
                    turn[index], winner[index], h[index], v[index]};
        std::vector<int> legal = board.legal_actions();
        if (cursor + static_cast<int>(legal.size()) > action_capacity)
            return -(cursor + static_cast<int>(legal.size()));
        std::copy(legal.begin(), legal.end(), actions + cursor);
        cursor += static_cast<int>(legal.size());
        offsets[index + 1] = cursor;
    }
    return cursor;
}

extern "C" void* bz_search_create(
    int n, int count, const int* p0, const int* p1,
    const uint64_t* h, const uint64_t* v, const int* w0, const int* w1,
    const int* turn, const int* winner, double c_puct, int* roots) {
    assert(n <= 9);
    NativeSearch* search = new NativeSearch{};
    search->n = n;
    search->action_size = 8 + 2 * (n - 1) * (n - 1);
    search->c_puct = c_puct;
    search->nodes.reserve(count * 64);
    for (int index = 0; index < count; ++index) {
        Board board{n, {p0[index], p1[index]}, {w0[index], w1[index]},
                    turn[index], winner[index], h[index], v[index]};
        roots[index] = search->add_node(board);
    }
    return search;
}

extern "C" void bz_search_destroy(void* handle) {
    delete static_cast<NativeSearch*>(handle);
}

int bz_search_prepare_pending(
    NativeSearch* search, float* encoded, int encoded_capacity,
    int* offsets, int* actions, int action_capacity) {
    auto started = std::chrono::steady_clock::now();
    int per_state = 8 * search->n * search->n;
    int evaluation_count = 0;
    int action_cursor = 0;
    offsets[0] = 0;
    for (NativePending& item : search->pending) {
        if (item.terminal) {
            item.evaluation_index = -1;
            continue;
        }
        if ((evaluation_count + 1) * per_state > encoded_capacity) {
            return -(evaluation_count + 1) * per_state;
        }
        const Board& board = search->nodes[item.leaf].board;
        int canonical_p0 = board.p[0];
        int canonical_p1 = board.p[1];
        uint64_t canonical_h = board.h;
        uint64_t canonical_v = board.v;
        int canonical_w0 = board.walls[0];
        int canonical_w1 = board.walls[1];
        if (board.turn == 1) {
            int cells = board.n * board.n;
            canonical_p0 = cells - 1 - board.p[1];
            canonical_p1 = cells - 1 - board.p[0];
            canonical_h = rotate_anchor_mask(board.h, board.n - 1);
            canonical_v = rotate_anchor_mask(board.v, board.n - 1);
            canonical_w0 = board.walls[1];
            canonical_w1 = board.walls[0];
        }
        auto encoding_started = std::chrono::steady_clock::now();
        int encoded_count = bz_encode_state_f32(
            board.n, canonical_p0, canonical_p1, canonical_h, canonical_v,
            canonical_w0, canonical_w1,
            encoded + evaluation_count * per_state, per_state
        );
        if (encoded_count != per_state) return -(evaluation_count + 1) * per_state;
        search->encoding_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - encoding_started
        ).count();

        auto legality_started = std::chrono::steady_clock::now();
        std::vector<int> legal = board.legal_actions();
        search->legality_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - legality_started
        ).count();
        if (action_cursor + static_cast<int>(legal.size()) > action_capacity)
            return -(action_cursor + static_cast<int>(legal.size()));
        std::copy(legal.begin(), legal.end(), actions + action_cursor);
        action_cursor += static_cast<int>(legal.size());
        offsets[evaluation_count + 1] = action_cursor;
        item.evaluation_index = evaluation_count++;
    }
    search->preparation_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return evaluation_count;
}

extern "C" int bz_search_prepare_roots(
    void* handle, const int* roots, int count,
    float* encoded, int encoded_capacity,
    int* offsets, int* actions, int action_capacity) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    search->set_root_pending(roots, count);
    return bz_search_prepare_pending(
        search, encoded, encoded_capacity, offsets, actions, action_capacity
    );
}

extern "C" int bz_search_descend_prepare(
    void* handle, const int* roots, int count,
    float* encoded, int encoded_capacity,
    int* offsets, int* actions, int action_capacity) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    search->descend(roots, count);
    return bz_search_prepare_pending(
        search, encoded, encoded_capacity, offsets, actions, action_capacity
    );
}

extern "C" int bz_search_expand_backup(
    void* handle, const float* policies, const float* values, int evaluation_count,
    const int* offsets, const int* actions) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    auto started = std::chrono::steady_clock::now();
    for (NativePending& item : search->pending) {
        if (item.terminal) {
            search->backup(item, item.terminal_value);
            continue;
        }
        int evaluation = item.evaluation_index;
        if (evaluation < 0 || evaluation >= evaluation_count) return -1;
        NativeNode& node = search->nodes[item.leaf];
        if (!node.expanded) {
            int begin = offsets[evaluation], end = offsets[evaluation + 1];
            double total = 0.0;
            for (int cursor = begin; cursor < end; ++cursor) {
                int action = actions[cursor];
                int policy_action = node.board.turn == 0
                    ? action : rotate_action_180(action, search->n);
                total += std::max(0.0, static_cast<double>(
                    policies[evaluation * search->action_size + policy_action]
                ));
            }
            node.edges.reserve(end - begin);
            for (int cursor = begin; cursor < end; ++cursor) {
                int action = actions[cursor];
                int policy_action = node.board.turn == 0
                    ? action : rotate_action_180(action, search->n);
                double prior = total > 0.0
                    ? std::max(0.0, static_cast<double>(
                        policies[evaluation * search->action_size + policy_action]
                    )) / total
                    : 1.0 / (end - begin);
                node.edges.push_back(NativeEdge{action, prior});
            }
            node.expanded = true;
            ++search->expansions;
        }
        search->backup(item, values[evaluation]);
    }
    search->expansion_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return 0;
}

extern "C" int bz_search_root_actions(
    void* handle, const int* roots, int count,
    int* offsets, int* actions, int capacity) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    int cursor = 0;
    offsets[0] = 0;
    for (int root_index = 0; root_index < count; ++root_index) {
        const NativeNode& root = search->nodes[roots[root_index]];
        if (cursor + static_cast<int>(root.edges.size()) > capacity)
            return -(cursor + static_cast<int>(root.edges.size()));
        for (const NativeEdge& edge : root.edges) actions[cursor++] = edge.action;
        offsets[root_index + 1] = cursor;
    }
    return cursor;
}

extern "C" int bz_search_add_noise(
    void* handle, const int* roots, int count,
    const int* offsets, const double* noise, double fraction) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    for (int root_index = 0; root_index < count; ++root_index) {
        NativeNode& root = search->nodes[roots[root_index]];
        int begin = offsets[root_index], end = offsets[root_index + 1];
        if (end - begin != static_cast<int>(root.edges.size())) return -1;
        for (int index = 0; index < end - begin; ++index) {
            root.edges[index].prior = (1.0 - fraction) * root.edges[index].prior
                + fraction * noise[begin + index];
        }
    }
    return 0;
}

extern "C" int bz_search_results(
    void* handle, const int* roots, int count, double temperature,
    double* policies, int* visits, double* priors, int output_count) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    int required = count * search->action_size;
    if (output_count < required) return -required;
    std::fill(policies, policies + required, 0.0);
    std::fill(visits, visits + required, 0);
    std::fill(priors, priors + required, 0.0);
    for (int root_index = 0; root_index < count; ++root_index) {
        const NativeNode& root = search->nodes[roots[root_index]];
        int row = root_index * search->action_size;
        for (const NativeEdge& edge : root.edges) {
            visits[row + edge.action] = edge.visits;
            priors[row + edge.action] = edge.prior;
        }
        if (temperature == 0.0) {
            int best = 0;
            for (int index = 1; index < static_cast<int>(root.edges.size()); ++index)
                if (root.edges[index].visits > root.edges[best].visits) best = index;
            policies[row + root.edges[best].action] = 1.0;
        } else if (temperature > 0.0) {
            std::vector<double> weights;
            weights.reserve(root.edges.size());
            double total = 0.0;
            for (const NativeEdge& edge : root.edges) {
                double weight = std::pow(edge.visits, 1.0 / temperature);
                weights.push_back(weight);
                total += weight;
            }
            if (total == 0.0) {
                for (const NativeEdge& edge : root.edges) total += edge.prior;
                for (const NativeEdge& edge : root.edges)
                    policies[row + edge.action] = edge.prior / total;
            } else {
                for (int index = 0; index < static_cast<int>(root.edges.size()); ++index)
                    policies[row + root.edges[index].action] = weights[index] / total;
            }
        } else {
            return -1;
        }
    }
    return required;
}

extern "C" int bz_search_advance(
    void* handle, const int* roots, const int* actions, int count,
    int* advanced, int* reused_visits) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    for (int root_index = 0; root_index < count; ++root_index) {
        int root_id = roots[root_index];
        int edge_index = -1;
        for (int index = 0; index < static_cast<int>(search->nodes[root_id].edges.size()); ++index)
            if (search->nodes[root_id].edges[index].action == actions[root_index]) {
                edge_index = index;
                break;
            }
        if (edge_index < 0) return -1;
        int child = search->nodes[root_id].edges[edge_index].child;
        if (child < 0) {
            Board child_board = search->nodes[root_id].board.apply_known_action(
                actions[root_index]
            );
            child = search->add_node(child_board);
            search->nodes[root_id].edges[edge_index].child = child;
        }
        advanced[root_index] = child;
        reused_visits[root_index] = search->nodes[child].total_visits;
    }
    return 0;
}

extern "C" int bz_search_get_states(
    void* handle, const int* node_ids, int count,
    int* p0, int* p1, uint64_t* h, uint64_t* v,
    int* w0, int* w1, int* turn, int* winner) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    for (int index = 0; index < count; ++index) {
        const Board& board = search->nodes[node_ids[index]].board;
        p0[index] = board.p[0]; p1[index] = board.p[1];
        h[index] = board.h; v[index] = board.v;
        w0[index] = board.walls[0]; w1[index] = board.walls[1];
        turn[index] = board.turn; winner[index] = board.winner;
    }
    return 0;
}

extern "C" int bz_search_stats(
    void* handle, long long* integers, double* seconds) {
    NativeSearch* search = static_cast<NativeSearch*>(handle);
    integers[0] = static_cast<long long>(search->nodes.size());
    integers[1] = search->expansions;
    integers[2] = search->descents;
    seconds[0] = search->traversal_seconds;
    seconds[1] = search->preparation_seconds;
    seconds[2] = search->expansion_seconds;
    seconds[3] = search->encoding_seconds;
    seconds[4] = search->legality_seconds;
    return 0;
}
