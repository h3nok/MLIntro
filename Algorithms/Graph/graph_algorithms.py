def breadth_first_search(graph, start):
    """1. First, we pop the first node from the queue and choose that as current node of
        this iteration.
            node = queue.pop(0)
        2. Then, we check that the node is not in the visited list. If it is not, we add it to the
        list of visited nodes and use neighbors to represent its directly connected nodes
            visited.append(node)
            neighbours = graph[node]
        3. Now we will add neighbours of nodes to the queue:
            for neighbour in neighbours:
                queue.append(neighbour)
        4. Once the main loop is complete, the visited data structure is returned, which
        contains all the nodes traversed.

    Args:
        graph (dict): input graph 
    """
    assert isinstance(graph, dict)

    visited = []
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]
            for neighbor in neighbours:
                queue.append(neighbor)
    return visited

def depth_first_search(graph, start, visited=None):
    """DFS is the alternative to BFS, used to search data from a graph. The factor that differentiates
       DFS from BFS is that after starting from the root vertex, the algorithm goes down as far as
       possible in each of the unique single paths one by one. For each path, once it has
       successfully reached the ultimate depth, it flags all the vertices associated with that path as
       visited. After completing the path, the algorithm backtracks. If it can find another path from
       the root node that has yet to be visited, the algorithm repeats the previous process. The
       algorithm keeps on moving in the new branch until all the branches have been visited.

    Args:
        graph ([type]): [description]
        start ([type]): [description]
        visited ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    assert isinstance(graph, dict)
    if visited is None:
        visited = set()
    visited.add(start)

    for next in graph[start] - visited:
        depth_first_search(graph, next, visited)

    return visited

graph={ 'Amin' : {'Wasim', 'Nick', 'Mike'},
'Wasim' : {'Imran', 'Amin'},
'Imran' : {'Wasim','Faras'},
'Faras' : {'Imran'},
'Mike' :{'Amin'},
'Nick' :{'Amin'}}

print(breadth_first_search(graph, 'Amin'))
print(depth_first_search(graph, 'Amin'))