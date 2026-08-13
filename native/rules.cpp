#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

namespace {
struct Board {
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
        std::deque<int> q{p[player]};
        std::vector<uint8_t> seen(n * n);
        seen[p[player]] = 1;
        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};
        while (!q.empty()) {
            int cell = q.front(); q.pop_front();
            int r = cell / n, c = cell % n;
            if (r == goal) return true;
            for (int i = 0; i < 4; ++i) {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
                int next = nr * n + nc;
                if (!seen[next] && !blocked(cell, next, hm, vm)) {
                    seen[next] = 1; q.push_back(next);
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
    template <typename T>
    void distance_plane(int player, T* output) const {
        int cells = n * n;
        std::vector<int> distance(cells, -1);
        std::deque<int> queue;
        int goal = player == 0 ? 0 : n - 1;
        for (int col = 0; col < n; ++col) {
            int cell = goal * n + col;
            distance[cell] = 0;
            queue.push_back(cell);
        }
        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};
        while (!queue.empty()) {
            int cell = queue.front(); queue.pop_front();
            int row = cell / n, col = cell % n;
            for (int i = 0; i < 4; ++i) {
                int nr = row + dr[i], nc = col + dc[i];
                if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
                int next = nr * n + nc;
                if (distance[next] == -1 && !blocked(cell, next, h, v)) {
                    distance[next] = distance[cell] + 1;
                    queue.push_back(next);
                }
            }
        }
        T scale = static_cast<T>(std::max(1, cells - 1));
        for (int cell = 0; cell < cells; ++cell)
            output[cell] = std::max(0, distance[cell]) / scale;
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
    if (winner >= 0) return 0;
    Board b{n, {p0,p1}, {w0,w1}, turn, winner, h, v};
    std::vector<int> actions;
    b.pawn_actions(actions);
    if (b.walls[turn] > 0) {
        int width = n - 1;
        for (int orientation = 0; orientation < 2; ++orientation)
            for (int r = 0; r < width; ++r)
                for (int c = 0; c < width; ++c) {
                    if (!b.geometry(orientation == 0, r, c)) continue;
                    int extra = orientation + 1;
                    if (b.path(0, extra, r, c) && b.path(1, extra, r, c))
                        actions.push_back(8 + orientation * width * width + r * width + c);
                }
    }
    if (static_cast<int>(actions.size()) > capacity) return -static_cast<int>(actions.size());
    std::copy(actions.begin(), actions.end(), output);
    return static_cast<int>(actions.size());
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
    board.distance_plane(0, output + 6 * cells);
    board.distance_plane(1, output + 7 * cells);
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
    board.distance_plane(0, output + 6 * cells);
    board.distance_plane(1, output + 7 * cells);
    return 8 * cells;
}
