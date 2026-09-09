from collections import deque

graph = {
    10: [20, 30],
    20: [40, 50],
    30: [60, 70],
    40: [],
    50: [80],
    60: [],
    70: [],
    80: []
}

def bfs_traversal(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    print("BFS Traversal Order:", end=" ")

    while queue:
        current = queue.popleft()
        print(current, end=" ")

        for neighbour in graph.get(current, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

print("Following is the Breadth-First Search")
bfs_traversal(graph, 10)