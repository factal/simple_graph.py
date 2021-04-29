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

class Vertex:
    def __init__(self, graph, index: int, content=None):
        self.graph = graph # 頂点が属するグラフ
        self.index = index # グラフ内でのインデックス番号, 隣接行列の構築などに用いる
        self.content = content # 頂点が保持する情報
        self.adj_vertices = [] # 隣接する頂点 
        self.connected_edges = [] # 頂点と繋がっている辺 (始点となる辺のみ保持)
    
    def __str__(self):
        return str(self.index)

class Edge:
    def __init__(self, graph, start: Vertex, end: Vertex, weight):
        self.graph = graph # 辺が属するグラフ
        self.start = start # 始点
        self.end = end # 終点
        self.weight = weight # 重み
        # self.adj_edges = [] # 隣接辺, 隣接情報の更新の際の計算量に関する問題を解決できなかったので現状サポートしていません
    
    def __str__(self):
        representation = '(' + str(self.start.index) + ', ' + str(self.end.index) +'): '+ str(self.weight)
        return representation

class Graph:
    def __init__(self, size: int=0, directed: bool=False): # size: 頂点数
        self.vertices = [Vertex(self, i) for i in range(size)] # 頂点集合
        self.edges = {} # 辺集合, {(始点, 終点): Edge, ...}
        # self.adj_matrix = [] # 隣接行列, メモリのオーバーヘッドを考慮してその都度 Graph.generate_adj_matrix() で生成するようにしました
        self._vertices_size = size # 頂点集合の大きさ
        self.directed = directed # 有向グラフかどうか
    
    def add_vertex(self, content=None) -> None:
        # 頂点の追加

        self.vertices.append(Vertex(self, self._vertices_size, content))
        self._vertices_size += 1
    
    def add_edge_from_index(self, start_index: int, end_index: int, weight: float=1) -> None:
        start_vertex = self.vertices[start_index]
        end_vertex = self.vertices[end_index]

        self.add_edge(start_vertex, end_vertex, weight)
    
    def add_edge(self, start: Vertex, end: Vertex, weight: float=1) -> None:
        # 辺の追加
        # Vertex.adj_vertices, Vertex.connected_edges も更新します

        new_edge = Edge(self, start, end, weight) 

        self.edges.setdefault((start, end), [])
        self.edges[(start, end)].append(new_edge)

        # 接続情報更新 (計算量削減および良い実装が思いつかなかったため、クラスメソッドに分けずここに直接書いています)
        start.adj_vertices.append(end)
        start.connected_edges.append(new_edge)

        # 無向グラフ用
        if self.directed:
            edge_reversed = Edge(self, end, start, weight)
            end.adj_vertices.append(start)
            self.edges.setdefault((end, start), [])
            self.edges[(end, start)].append(edge_reversed)
            end.connected_edges.append(edge_reversed)


    def get_weight_from_vertices(self, start: Vertex, end: Vertex) -> float:
        # Vertex の組から重みを取得, 多重辺が存在する場合は最小の重みを返す
        # O(E)?
    
        connected_edges = self.edges[(start, end)]
        return min([e.weight for e in connected_edges])
    
    def get_connected_vertices(self, new_edge: Edge) -> tuple:
        # Edge に接続された Vertex を (始点, 終点) のタプルで返す
        
        return (self.vertices[new_edge.start], self.vertices[new_edge.end])
    
    def generate_adj_matrix(self) -> list:
        # 隣接行列生成
        # O(E)?

        adj_matrix = [[0] * self._vertices_size for _ in range(self._vertices_size)] # init

        for start, end in self.edges.keys():
            start_index = start.index
            end_index = end.index

            adj_matrix[start_index][end_index] = self.get_weight_from_vertices(start, end)
        
        return adj_matrix



# --- algorithms --- #



def dfs(graph: Graph, root: Vertex, visit_all_vertices=False) -> dict:
    # O(V+E)
    # visit_all_vertices: すべての頂点を探索するかどうか (True のとき pred は深さ優先木探索結果の深さ優先森を記録する)
    # 入力: グラフ graph, 探索の始点 root
    # 出力: 深さ優先順序において前の頂点を記録した辞書 pred
    
    def __dfs_visit(current: Vertex) -> None:
        label[current] = 'searching'
        for next in current.adj_vertices:
            if label[next] == 'unsearched': # 未訪問地点を見つけ、その方向に進む
                pred[next] = current
                __dfs_visit(next)
        label[current] = 'searched' 
    
    # init
    pred = {v: None for v in graph.vertices}
    label = {v: 'unsearched' for v in graph.vertices}
    
    if visit_all_vertices:
        for key, state in label.items():
            if state == 'unsearched': 
                __dfs_visit(key)

    else:
        __dfs_visit(root)
    
    return pred
    
def bfs(graph: Graph, root: Vertex) -> tuple:
    # O(V+E)
    # 入力: グラフ graph, 探索の始点 root
    # 出力: (幅優先順序において前の頂点を記録した辞書 pred, 始点からの距離の辞書 distance)
    
    # init
    pred = {v: None for v in graph.vertices}
    label = {v: 'unsearched' for v in graph.vertices}
    distance = {v: float('inf') for v in graph.vertices}
    
    label[root] = 'searching'
    distance[root] = 0
    visited = deque([root])
    
    while len(visited) != 0:
        current = visited.popleft()
        for next in current.adj_vertices:
            if label[next] == 'unsearched':
                pred[next] = current
                label[next] = 'searching'
                distance[next] = distance[current] + 1

                visited.append(next)
        label[current] = 'searched'
    
    return (pred, distance)

def dijkstra(graph: Graph, root) -> tuple:
    # single source shortest path, O((V+E)*logV)
    # 入力: 有向重み付きグラフ graph (任意の重みは非負), 始点 root
    # 出力: (始点から各頂点に至る実際の最短経路の再計算に用いる辞書 pred, 始点からの距離の辞書 distance)
    
    # init
    pred = {v: None for v in graph.vertices}
    distance = {v: float('inf') for v in graph.vertices}
    distance[root] = 0
    
    queue = pq()
    for v in graph.vertices:
        queue.push(v, distance[v]) # 最短経路距離を優先度として全頂点を優先度付きキュー queue に push
    
    while len(queue) != 0:
        current = queue.pop() # 始点への最短経路の頂点を取り出す

        for next in current.adj_vertices:
            length = distance[current] + graph.get_weight_from_vertices(current, next)

            if length < distance[next]:
                # current から next へのより短い経路で queue を更新し記録

                queue.set_priority(next, length)
                distance[next] = length
                pred[next] = current
    
    return (pred, distance)

def bellman_ford(graph: Graph, root: Vertex) -> tuple:
    # single source shortest path, O(V*E)
    # 入力: 有向重み付きグラフ graph (重みの総和が負となる閉路が存在してはならない), 始点 root
    # 出力: (始点から各頂点に至る実際の最短経路の再計算に用いる辞書 pred, 始点からの距離の辞書 distance)
    
    # init
    pred = {v: None for v in graph.vertices}
    distance = {v: float('inf') for v in graph.vertices}
    distance[root] = 0
    
    for i in range(len(graph.vertices)):
        for e in graph.edges:
            current, next = graph.get_connected_vertices(e)
            length = distance[current] + graph.get_weight_from_vertices(current, next)

            if length < distance[next]:
                # current から next へのより短い経路を記録

                if i == len(graph.vertices):
                    # Bellman Ford 法は |V|-1 回の走査で最短経路を算出できるため、|V| 回目の経路の更新は重みの総和が負の閉路が存在することを意味する
                    print('found a cycle with negative weight')
                
                distance[next] = length
                pred[next] = current
    
    return (pred, distance)
            
def warshall_floyd(graph: Graph) -> tuple:
    # all pair shortest path, O(V^3)
    # 入力: 有向重み付きグラフ graph (任意の重みは正)
    # 出力: (各頂点間の最短経路の再計算に用いる2次元辞書 pred, 各頂点から全頂点への距離の2次元辞書 distance)
    
    # init
    pred = {}
    distance = {}
    
    for current in graph.vertices:
        pred[current] = {v: None for v in graph.vertices}
        distance[current] = {v: float('inf') for v in graph.vertices}
        distance[current][current] = 0
        
        for next in current.adj_vertices:
            distance[current][next] = graph.get_weight_from_vertices(current, next)
            pred[current][next] = current
        
    for k in graph.vertices:
        for u in graph.vertices:
            for v in graph.vertices:
                length = distance[u][k] + distance[k][v]
                if length < distance[u][v]:
                    # current から next へのより短い経路を記録

                    distance[u][v] = length
                    pred[u][v] = pred[k][v] # 新たに得られた1つ前のリンクを記録
    
    return (pred, distance)

def prim(graph: Graph, root: Vertex) -> dict:
    # MST, O((V+E)logV)
    # 入力: 無向グラフ graph, 探索を開始する頂点 root
    # 出力: pred に収められたMST (MSTの根は -1)
    
    # init
    pred = {v: None for v in graph.vertices}
    pred[root] = -1
    distance = {v: float('inf') for v in graph.vertices}
    distance[root] = 0
    
    queue = pq()
    for v in graph.vertices:
        queue.push(v, distance[v])
    
    while len(queue) != 0:
        current = queue.pop() # 計算された距離が最小の頂点を取得
        for next in current.adj_vertices:
            weight = graph.get_weight_from_vertices(current, next)
            if weight < distance[next]:
                pred[next] = current
                distance[next] = weight # next の推定コスト更新
                queue.set_priority(current, weight)
    
    return pred

def make_mst(original: Graph, pred: dict) -> Graph:
    # prim() で得た pred から MST の graph を作成
    
    mst = Graph(len(original.vertices), directed=False) # init
    
    for end, start in pred.items():
        if isinstance(start, Vertex):
            weight = original.get_weight_from_vertices(start, end)
            mst.add_edge_from_index(start.index, end.index, weight)
    
    return mst

# --- テストコード --- #

test_graph = Graph(5, directed=True) # 5頂点のグラフを作成
test_edges = [[0, 1, 2], [0, 4, 4], [1, 2, 3], [2, 4, 1], [2, 3, 5],  [3, 0, 8], [3, 4, 7], [4, 3, 7]] # 辺集合
# 辺を追加
for e in test_edges:
    test_graph.add_edge_from_index(e[0], e[1], e[2])

print('デモ: prim()')
print('隣接行列:\n', test_graph.generate_adj_matrix(), '\n', sep='')

pred = prim(test_graph, test_graph.vertices[0])
MST = make_mst(test_graph, pred)

print('最小被覆木:\n', MST.generate_adj_matrix(), '\n', sep='')

test_graph = Graph(5, directed=False)
for e in test_edges:
    test_graph.add_edge_from_index(e[0], e[1], e[2])

print('デモ: dijkstra()')
print('隣接行列:\n', test_graph.generate_adj_matrix(), '\n', sep='')

prec, dist = dijkstra(test_graph, test_graph.vertices[0])

for dest, d in dist.items():
    print('to v_' + str(dest) + ': ' + str(d))
