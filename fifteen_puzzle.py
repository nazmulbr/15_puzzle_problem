import random
import math
import heapq
from collections import deque
from typing import Tuple, List, Dict, Optional, Set
from functools import lru_cache

N = 4
GOAL: Tuple[int, ...] = tuple(range(1, N*N)) + (0,)


def pretty(state: Tuple[int, ...]) -> str:
    rows = []
    for i in range(0, N*N, N):
        row = state[i:i+N]
        rows.append(" ".join("{:>2}".format(
            x if x != 0 else "_") for x in row))
    return "\n".join(rows)


def get_moves(pos: int) -> List[int]:
    x, y = divmod(pos, N)
    moves = []
    if x > 0:
        moves.append(pos - N)
    if x < N - 1:
        moves.append(pos + N)
    if y > 0:
        moves.append(pos - 1)
    if y < N - 1:
        moves.append(pos + 1)
    return moves


def apply_move(state: Tuple[int, ...], blank: int, new_pos: int) -> Tuple[Tuple[int, ...], int]:
    s = list(state)
    s[blank], s[new_pos] = s[new_pos], s[blank]
    return tuple(s), new_pos


def is_solvable(state: Tuple[int, ...]) -> bool:
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    blank_row_from_top = state.index(0) // N
    if N % 2 == 1:
        return inv % 2 == 0
    else:
        blank_row_from_bottom = (N - blank_row_from_top)
        return (inv + blank_row_from_bottom) % 2 == 1


def manhattan(state: Tuple[int, ...]) -> int:
    dist = 0
    for idx, val in enumerate(state):
        if val == 0:
            continue
        gx = (val - 1) // N
        gy = (val - 1) % N
        x = idx // N
        y = idx % N
        dist += abs(gx - x) + abs(gy - y)
    return dist


def scramble_from_goal(steps: int, seed: Optional[int] = None) -> Tuple[int, ...]:
    rng = random.Random(seed)
    state = GOAL
    blank = state.index(0)
    prev = None
    for _ in range(steps):
        mvs = get_moves(blank)
        if prev is not None and prev in mvs and len(mvs) > 1:
            mvs.remove(prev)
        mv = rng.choice(mvs)
        new_state, new_blank = apply_move(state, blank, mv)
        prev, state, blank = blank, new_state, new_blank
    return state

# --- DFS Backtracking ---


def dfs_backtracking(initial: Tuple[int, ...], depth_limit: int = 22):
    visited: Set[Tuple[int, ...]] = set([initial])
    path: List[Tuple[int, ...]] = [initial]
    nodes = 0

    def rec(state: Tuple[int, ...], blank: int, depth: int) -> bool:
        nonlocal nodes
        nodes += 1
        if state == GOAL:
            return True
        if depth >= depth_limit:
            return False
        for mv in get_moves(blank):
            ns, nb = apply_move(state, blank, mv)
            if ns in visited:
                continue
            visited.add(ns)
            path.append(ns)
            if rec(ns, nb, depth+1):
                return True
            path.pop()
            visited.remove(ns)
        return False
    found = rec(initial, initial.index(0), 0)
    return found, path if found else [], nodes

# --- DP with Memoization ---


def dp_min_steps(initial: Tuple[int, ...], max_depth: int = 20):
    nodes = 0

    @lru_cache(maxsize=None)
    def rec(state: Tuple[int, ...], depth_left: int) -> int:
        nonlocal nodes
        nodes += 1
        if state == GOAL:
            return 0
        if depth_left < 0:
            return math.inf
        best = math.inf
        blank = state.index(0)
        for mv in get_moves(blank):
            ns, _ = apply_move(state, blank, mv)
            cost = rec(ns, depth_left - 1)
            if cost != math.inf:
                best = min(best, 1 + cost)
        return best
    res = rec(initial, max_depth)
    return -1 if res == math.inf else int(res), nodes

# --- BFS Shortest Path ---


def bfs_shortest_path(initial: Tuple[int, ...]):
    q = deque([initial])
    parent: Dict[Tuple[int, ...], Optional[Tuple[int, ...]]] = {initial: None}
    nodes = 0
    while q:
        s = q.popleft()
        nodes += 1
        if s == GOAL:
            path = []
            cur = s
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return len(path)-1, path, nodes
        blank = s.index(0)
        for mv in get_moves(blank):
            ns, _ = apply_move(s, blank, mv)
            if ns not in parent:
                parent[ns] = s
                q.append(ns)
    return -1, [], nodes

# --- A* Search ---


def astar(initial: Tuple[int, ...]):
    g: Dict[Tuple[int, ...], int] = {initial: 0}
    parent: Dict[Tuple[int, ...], Optional[Tuple[int, ...]]] = {initial: None}
    pq = [(manhattan(initial), 0, initial)]
    nodes = 0
    while pq:
        f, g_cost, s = heapq.heappop(pq)
        nodes += 1
        if s == GOAL:
            path = []
            cur = s
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return g_cost, path, nodes
        blank = s.index(0)
        for mv in get_moves(blank):
            ns, _ = apply_move(s, blank, mv)
            ng = g_cost + 1
            if ns not in g or ng < g[ns]:
                g[ns] = ng
                parent[ns] = s
                heapq.heappush(pq, (ng + manhattan(ns), ng, ns))
    return -1, [], nodes


if __name__ == "__main__":
    init = scramble_from_goal(5, seed=42)
    print("Initial state:\\n", pretty(init))
    print("Solvable:", is_solvable(init))

    print("\\n-- DFS (depth_limit=10) --")
    found, path, nodes = dfs_backtracking(init, depth_limit=10)
    print("found:", found, "steps:", len(path) -
          1 if found else -1, "nodes:", nodes)

    print("\\n-- DP (max_depth=10) --")
    steps, nodes_dp = dp_min_steps(init, max_depth=10)
    print("dp steps:", steps, "nodes:", nodes_dp)

    print("\\n-- BFS --")
    s_bfs, p_bfs, n_bfs = bfs_shortest_path(init)
    print("bfs steps:", s_bfs, "nodes:", n_bfs)

    print("\\n-- A* --")
    s_astar, p_astar, n_astar = astar(init)
    print("astar steps:", s_astar, "nodes:", n_astar)
