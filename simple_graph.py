from collections import deque
import heapq

class pq:
    # 優先度付きキュー　(priority queue)
    
    def __init__(self):
        self.data = [] # 要素を格納, (priority, index, item) 
        self.index = 0
    
    def __len__(self):
        return len(self.data)
    
    def push(self, item, priority):
        heapq.heappush(self.data, (priority, self.index, item))
        self.index += 1
    
    def pop(self):
        return heapq.heappop(self.data)[2]
        
    def set_priority(self, seek_item, priority):
        # pq内の特定の要素を探して優先度を変更する
        # 要素が互いに異なる場合しかうまく動作しません！
        
        for index, data in enumerate(self.data):
            if seek_item == data[2]: # item
                data = list(data)
                data[0] = priority
                self.data[index] = tuple(data)
                break

class vertex:
    def __init__(self, index: int, content=None):
        self.index = index # グラフ内でのインデックス
        self.content = content # 頂点が保持する情報
        self.adj = [] # 隣接する頂点
    
    def __str__(self):
        return str(self.index)
    
    def init_adj(self, graph):
        # 隣接行列 graph.matrix で隣接リスト vertex.adj を初期化
        
        for index_adj, edge in enumerate(graph.adj_matrix[self.index]):
            if self.index == index_adj:
                continue
            elif edge != float('inf'):
                self.adj.append(graph.vertexes[index_adj])

class edge:
    def __init__(self, start: vertex, end: vertex, weight=1):
        self.start = start # 始点
        self.end = end # 終点
        self.weight = weight # 重み

class graph:
    def __init__(self, size: int=0): # size: 頂点数
        self.adj_matrix = [[float('inf')] * size for _ in range(size)] # 隣接行列
        self.vertexes = [vertex(index) for index in range(size)] # 頂点集合
        self.edges = [] # 辺集合
    
    def __str__(self):
        return '\n'.join(map(str, self.adj_matrix))
    
    def init_edges(self, edges: list, one_index=False):
        # self.edges 更新
        # edges の凡例: [[始点のインデックス, 終点のインデックス, 重み], [始点のインデックス, 終点のインデックス, 重み], ...] または [[始点のインデックス, 終点のインデックス], [始点のインデックス, 終点のインデックス], ...]
        
        if len(edges[0]) == 2: # unweighted
            for start, end in edges:
                self.edges.append(edge(start - one_index, end - one_index))
        
        elif len(edges[0]) == 3: # weighted
            for start, end, weight in edges:
                self.edges.append(edge(start - one_index, end - one_index, weight))

    def update_connection_info(self, is_directed=False):
        # self.edges を元に頂点/辺の接続情報を更新
        # is_directed: Trueのとき 有向グラフ, Falseのとき 無向グラフ
        
        # 隣接行列更新
        if is_directed:
            for e in self.edges:
                self.adj_matrix[e.start][e.end] = e.weight        
        else:
            for e in self.edges:
                self.adj_matrix[e.start][e.end] = e.weight
                self.adj_matrix[e.end][e.start] = e.weight
        
        # vertex の隣接リスト更新
        for v in self.vertexes:
            v.init_adj(self)
    
    def get_weight(self, start: vertex, end: vertex) -> int:
        # vertex -> weight
        
        return self.adj_matrix[start.index][end.index]  
    
    def get_connected_vertexes(self, e: edge):
        # edge -> vertex
        
        return (self.vertexes[e.start], self.vertexes[e.end])
            
# --- algorithms --- #



def dfs(graph: graph, root: vertex, visit_all_vertexes=False) -> dict:
    # O(V+E)
    # visit_all_vertexes: すべての頂点を探索するかどうか (True のとき pred は深さ優先木探索結果の深さ優先森を記録する)
    # 入力: グラフ graph, 探索の始点 root
    # 出力: 深さ優先順序において前の頂点を記録した辞書 pred
    
    def _dfs_visit(current: vertex):
        label[current] = 'Searching'
        for next in current.adj:
            if label[next] == 'Unsearched':
                pred[next] = current
                _dfs_visit(next)
        label[current] = 'Searched' 
    
    pred = {v: None for v in graph.vertexes}
    label = {v: 'Unsearched' for v in graph.vertexes}
    
    if visit_all_vertexes:
        for key, state in label.items():
            if state == 'Unsearched':
                _dfs_visit(key)
    else:
        _dfs_visit(root)
    
    return pred
    
def bfs(graph: graph, root: vertex) -> tuple:
    # O(V+E)
    # 入力: グラフ graph, 探索の始点 root
    # 出力: (幅優先順序において前の頂点を記録した辞書 pred, 始点からの距離の辞書 distance)
    
    pred = {v: None for v in graph.vertexes}
    label = {v: 'Unsearched' for v in graph.vertexes}
    distance = {v: float('inf') for v in graph.vertexes}
    
    label[root] = 'Searching'
    distance[root] = 0
    visited = deque([root])
    
    while len(visited) != 0:
        current = visited.popleft()
        for next in current.adj:
            if label[next] == 'Unsearched':
                pred[next] = current
                label[next] = 'Searching'
                distance[next] = distance[current] + 1

                visited.append(next)
        label[current] = 'Searched'
    
    return (pred, distance)

def dijkstra(graph: graph, root) -> tuple:
    # single source shortest path, O((V+E)*logV)
    # 入力: 有向重み付きグラフ graph (任意の重みは非負), 始点 root
    # 出力: (始点から各頂点に至る実際の最短経路の再計算に用いる辞書 pred, 始点からの距離の辞書 distance)
    
    pred = {v: None for v in graph.vertexes}
    distance = {v: float('inf') for v in graph.vertexes}
    distance[root] = 0
    
    queue = pq()
    for v in graph.vertexes:
        queue.push(v, distance[v])
    
    while len(queue) != 0:
        current = queue.pop()
        for next in current.adj:
            length = distance[current] + graph.get_weight(current, next)
            if length < distance[next]:
                queue.set_priority(next, length)
                distance[next] = length
                pred[next] = current
    
    return (pred, distance)

def bellman_ford(graph: graph, root: vertex):
    # single source shortest path, O(V*E)
    # 入力: 有向重み付きグラフ graph (重みの総和が負となる閉路が存在してはならない), 始点 root
    # 出力: (始点から各頂点に至る実際の最短経路の再計算に用いる辞書 pred, 始点からの距離の辞書 distance)
    
    pred = {v: None for v in graph.vertexes}
    distance = {v: float('inf') for v in graph.vertexes}
    distance[root] = 0
    
    for i in range(len(graph.vertexes)):
        for e in graph.edges:
            current, next = graph.get_connected_vertexes(e)
            length = distance[current] + graph.get_weight(current, next)
            if length < distance[next]:
                if i == len(graph.vertexes):
                    print('found a cycle with negative weight')
                
                distance[next] = length
                pred[next] = current
    
    return (pred, distance)
            
def warshall_floyd(graph: graph):
    # all pair shortest path, O(V^3)
    # 入力: 有向重み付きグラフ graph (任意の重みは正)
    # 出力: (各頂点間の最短経路の再計算に用いる2次元辞書 pred, 各頂点から全頂点への距離の2次元辞書 distance)
    
    pred = {}
    distance = {}
    
    for current in graph.vertexes:
        pred[current] = {v: None for v in graph.vertexes}
        distance[current] = {v: float('inf') for v in graph.vertexes}
        distance[current][current] = 0
        
        for next in current.adj:
            distance[current][next] = graph.get_weight(current, next)
            pred[current][next] = current
        
    for k in graph.vertexes:
        for u in graph.vertexes:
            for v in graph.vertexes:
                length = distance[u][k] + distance[k][v]
                if length < distance[u][v]:
                    distance[u][v] = length
                    pred[u][v] = pred[k][v]
    
    return (pred, distance)

def prim(graph: graph, root: vertex) -> dict:
    # MST, O((V+E)logV)
    # 入力: 無向グラフ graph, 探索を開始する頂点 root
    # 出力: pred に収められたMST (MSTの根は -1)
    
    # init
    pred = {v: None for v in graph.vertexes}
    pred[root] = -1
    distance = {v: float('inf') for v in graph.vertexes}
    distance[root] = 0
    
    queue = pq()
    for v in graph.vertexes:
        queue.push(v, distance[v])
    
    while len(queue) != 0:
        current = queue.pop() # 距離が最小の頂点を取得
        for next in current.adj:
            weight = graph.get_weight(current, next)
            if weight < distance[next]:
                pred[next] = current
                distance[next] = weight # next の推定コスト更新
                queue.set_priority(current, weight)
    
    return pred

def make_mst(original: graph, pred: dict) -> graph:
    # prim で得た pred から MST の graph を作成
    
    mst = graph(len(original.vertexes))
    mst.vertexes = original.vertexes
    
    edges = []
    for end, start in pred.items():
        if isinstance(start, vertex):
            weight = original.get_weight(start, end)
            edges.append([start.index, end.index, weight])
    
    mst.init_edges(edges)
    mst.update_connection_info(is_directed=True)
    
    return mst

# --- テストコード --- #

test_graph = graph(5) # 5頂点のグラフを作成
test_edges = [[0, 1, 2], [0, 4, 4], [1, 2, 3], [2, 4, 1], [2, 3, 5],  [3, 0, 8], [3, 4, 7], [4, 3, 7]] # 辺
test_graph.init_edges(test_edges)
test_graph.update_connection_info(is_directed=False) # 辺情報更新

print('デモ: prim()')
print('隣接行列:\n', test_graph, '\n', sep='')

pred = prim(test_graph, test_graph.vertexes[0])
MST = make_mst(test_graph, pred)

print('最小被覆木:\n', MST, '\n', sep='')

test_graph = graph(5)
test_graph.init_edges(test_edges)
test_graph.update_connection_info(is_directed=True)

print('デモ: dijkstra()')
print('隣接行列:\n', test_graph, '\n', sep='')

prec, dist = dijkstra(test_graph, test_graph.vertexes[0])

for dest, d in dist.items():
    print('to v_' + str(dest) + ': ' + str(d))
