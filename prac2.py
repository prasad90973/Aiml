import heapq


def a_star(graph, heuristic, start, goal):

    open_list = [(0, start)]

    g_cost = {node: float("inf") for node in graph}
    g_cost[start] = 0

    parent = {node: None for node in graph}

    while open_list:

        current_f, current = heapq.heappop(open_list)

        if current == goal:
            path = []

            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            return path, g_cost[goal]

        for neighbor, cost in graph[current]:

            new_g_cost = g_cost[current] + cost

            if new_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_g_cost
                parent[neighbor] = current

                f_cost = new_g_cost + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))

    return None, float("inf")


graph = {
    "P": [("Q", 3), ("R", 6)],
    "Q": [("P", 3), ("R", 2), ("S", 7)],
    "R": [("P", 6), ("Q", 2), ("S", 3)],
    "S": [("Q", 7), ("R", 3), ("T", 4)],
    "T": [("S", 4)]
}


heuristic = {
    "P": 9,
    "Q": 7,
    "R": 5,
    "S": 3,
    "T": 0
}


start = "P"
goal = "T"


path, cost = a_star(graph, heuristic, start, goal)


if path:
    print("Optimal Path:", " -> ".join(path))
    print("Total Cost:", cost)
else:
    print("No path found.")