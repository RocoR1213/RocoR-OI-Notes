# RocoR's OI Note 第三章 搜索与图论

本笔记基于AcWing《算法基础课》第3章 搜索与图论

## 邻接矩阵

> **用途：**
>
> 存储稠密图
>
> **最短路问题初始化：**关注边的权值
>
> ```cpp
> const int INF = 0x3f3f3f3f;
> 
> for (int i = 1; i <= n; i++)
> 	for (int j = 1; j <= n; j++) {
> 		if (i == j) g[i][j] = 0;	// 自己到自己距离为0
> 		else g[i][j] = INF;			// 其他点默认不可达
> }
> ```
>
> **图的连通性判断初始化：**关注边的有无
>
> ```cpp
> for (int i = 1; i <= n; i++)
>     for (int j = 1; j <= n; j++)
>         g[i][j] = 0;	// 初始没有边
> ```
>
> * 此情景下若`g[i][i] == 1`意味着存在自环

## 邻接表

> **用途：**
> 存储稀疏图/树
>
> **模板：**
>
> ```cpp
> //（1）链式前向星
> 
> //邻接表存图：adj[u]指向一个链表的第一个结点，链表中存储从结点u出发的所有有向边(u -> v)中的v
> int adj[N];	
> int e[N], ne[N], idx = 1;	//链表的实现（见第二章模拟单链表）
> 
> // 初始化邻接表，所有结点指向-1，代表空
> memset(adj, -1, sizeof adj);
> 
> // --------------------------------基本操作--------------------------------
> 
> // 向图中添加一条边 a -> b，链表头插法
> void add(int a, int b){
>        e[idx] = b;
>        ne[idx] = adj[a];
>        adj[a] = idx++;
> }
> 
> // 枚举结点 t 的所有邻接点（从 t 能走到的所有结点）
> for (int i = adj[t]; i != -1; i = ne[i]){
>        //......
> }
> 
> 
> //（2）STL vector邻接表
> 
> // 邻接表存图：adj[u]对应一个数组，其中存储从结点u出发的所有有向边(u -> v)中的v
> vector<vector<int>> adj(N);
> 
> // --------------------------------基本操作--------------------------------
> 
> // 向有向图中添加一条边 a -> b
> void add(int a, int b){
>        adj[a].push_back(b);
> }
> 
> // 枚举结点 t 的所有邻接点（从 t 能走到的所有结点）
> for (auto neighbor : adj[t]){
>        //......
> }
> ```



## 深度优先搜索DFS

>  DFS 有一个**通用的逻辑结构**。可以把所有 DFS 代码看作由以下 **4 个部分** 组成的：
>
> 1. **截止条件 (Base Case)**：什么时候停下来？（比如：走到终点了、越界了、或者数字填满了）。
> 2. **当前层逻辑 (Process)**：到了这个节点，我要做什么？（比如：标记我已经来过这里，或者把当前数字加入答案）。
> 3. **下探 (Recursion)**：根据当前状态，尝试所有可能的下一步。
> 4. **回溯 (Backtracking - \*可选\*)**：从深层回来后，要不要恢复现场？
>
> **思路伪代码：**
>
> ```cpp
> void dfs(当前状态参数) {
>      // 1. 截止条件（递归出口）
>      if (到达终点 || 状态不合法) {
>          更新答案;
>          return;
>      }
> 
>      // 2. 遍历所有可能的“下一步”
>      for (每一个可能的选择 in current_options) {
>         
>          // --- 做出选择 ---
>          标记当前选择 (visited[next] = true);
>          记录路径 (path.push(next));
> 
>          // 3. 下探（进入下一层递归）
>          dfs(下个状态参数);
> 
>          // 4. 回溯（恢复现场 - 非常关键！）
>          // 只有在需要“多种组合”或“多次使用同一资源”时才需要这一步
>          撤销标记 (visited[next] = false);
>          撤销路径 (path.pop());
>          // --- 撤销选择 ---
>      }
> }
> ```
>
> 



## 宽度优先搜索BFS

> **模板：**
>
> ```cpp
> // 以最短路问题为例
> // 1. 定义队列 + 距离/访问数组
> queue<int> q;	// 队列，用于存放待扩展的结点（BFS 的核心数据结构）
> vector<int> dist(N, -1);// -1 表示该结点没访问过
> 
> // 2. 初始化起点距离，起点入队
> dist[start] = 0;
> q.push(start);
> 
> // 3. BFS 主循环
> while (!q.empty()) {
>        auto t = q.front();
>        q.pop();
> 
>        // 4. 枚举 t 能走到的所有下一步 next
>        for (所有 next from t) {
>            if (next 合法 && dist[next] == -1) {
>                dist[next] = dist[t] + 1;
>                q.push(next);
>            }
>        }
> }
> 
> // dist[target] 就是答案（如果target不可达，返回-1）
> 
> ```
>
> 

## 拓扑排序（Kahn算法）

> ```cpp
> int q[N], hh = 0, tt = -1;		// q: 模拟队列，存储入度为0的节点
> int d[N];						// d: 存储每个节点的入度 (in-degree)
> 
> // 邻接表存储图：adj[x] 包含所有从 x 出发能到达的节点
> vector<vector<int>> adj(N);
> 
> // 拓扑排序核心函数
> bool topSort(){
>        // 1. 初始化：遍历所有节点（注意题目通常是1到n），将初始入度为0的点入队
>        for (int i = 1; i <= n; i++)
>            if (d[i] == 0)
>                q[++tt] = i; // 入队操作：队尾指针后移并存入值
> 
>        // 2. BFS 循环处理
>        while (hh <= tt){
>            // 出队操作：取出队头元素，队头指针后移
>            int t = q[hh++];
> 
>            // 遍历当前节点 t 的所有邻居 u (即存在边 t -> u)
>            for (auto neighbor : adj[t]){
>                int u = neighbor;
> 
>                // 核心逻辑：既然 t 已经处理完（被删去），u 的入度减 1
>                d[u]--;
> 
>                // 如果 u 的入度变为 0，说明 u 的所有前置条件都满足了，可以入队
>                if (d[u] == 0) q[++tt] = u;
>            }
>        }
> 
>        // 3. 判断结果
>        // 队列中存储了所有进入拓扑序列的点。
>        // 如果入队的点总数等于 n ，说明成功排好序。
>        // 如果小于 n，说明图中存在环，导致环上的点入度永远无法减为 0，也就无法入队。
>        return tt == n - 1;
> }
> ```
>
> **说明：**
>
> * **有向无环图**又称拓扑图
>
> * 有向无环图一定至少存在一个**入度**为0的点（反证法 + 抽屉原理证明）
>
> * 当循环结束后，若队列的长度为`n`，则拓扑排序存在
>
> * 在添加边时，可以顺便统计入度
>
> * 此处使用模拟队列，“弹出”队头时只是指针后移，并未删除元素，这样便于输出排序结果
>
> * 时间复杂度为$O(n+m)$

## Dijkstra

### 朴素Dijkstra算法

> 单源最短路算法
> 使用邻接矩阵存储边，适合稠密图
>
> **模板：**
>
> ```cpp
> int g[N][N];  // 邻接矩阵存储每条边
> int dist[N];  // 存储起点到每个点的当前最短距离
> bool st[N];   // 存储每个点的最短路是否已经确定
> 
> // 求1号点到n号点的最短路，如果不存在则返回-1
> int dijkstra()
> {
>     memset(dist, 0x3f, sizeof dist);        // 0x3f3f3f3f作为距离的“最大值”
>     dist[1] = 0;                            // 起点到自己的距离为0
> 
>     for (int i = 0; i < n - 1; i ++ )       // 执行n-1次（自己到自己的距离已经确定）
>     {
>         // 在还未确定最短路的点中，寻找距离最小的点
>         int t = -1;
>         for (int j = 1; j <= n; j ++ )
>                if (!st[j] && (t == -1 || dist[t] > dist[j]))
>                    t = j;
> 
>         // 标记该点最短路已确定
>         st[t] = true;
> 
>         // 用t更新其他点的距离（松弛）
>         for (int j = 1; j <= n; j ++ )
>                dist[j] = min(dist[j], dist[t] + g[t][j]);
>     }
> 
>     if (dist[n] == 0x3f3f3f3f) return -1;       // 不可达
>     return dist[n];
> }
> ```
>
> * 时间复杂度为$O(n^2)$

### 堆优化Dijkstra算法

> 使用邻接表+堆优化，适合稀疏图
>
> **模板：**
>
> ```cpp
> // 堆中结点和邻接表中的边均用二元组存储
> typedef pair<int, int> PII;
> 
> // 邻接表存储边
> // 格式：adj[u] = { {v1, w1}, {v2, w2}, {目标点, 权值}... }
> vector<vector<PII>> adj(N);
> int dist[N];
> bool st[N];
> 
> // 加边函数
> void addEdge(int x, int y, int w) {
>        adj[x].push_back({y, w});
> }
> 
> int dijkstra() {
>        memset(dist, 0x3f, sizeof dist);
>        // 定义优先队列（小根堆）
>        // 堆中存储格式：{距离, 节点编号},因为pair默认按first排序
>        priority_queue<PII, vector<PII>, greater<PII>> heap;
>        dist[1] = 0;
>        heap.push({dist[1], 1});
> 
>        while(!heap.empty()) {
>            // 结构化绑定，取出堆顶（当前距离源点最近的未确定点）
>            auto [d, u] = heap.top(); // d是当前点距离，u是当前点编号
>            heap.pop();
>     
>            if (st[u]) continue;
>            // 标记该点最短路已确定
>            st[u] = true;
>         
>         // 扫描t的所有邻居进行松弛操作
>            for (auto [v, w] : adj[u]) { // v是目标点编号，w是u->v这条边权值
>                if (dist[v] > d + w) {
>                    dist[v] = d + w;
>                    heap.push({dist[v], v}); // 将新产生的路径结点入堆
>                }
>            }
>     }
> 
>        if (dist[n] == 0x3f3f3f3f) return -1;
>        return dist[n];
> }
> ```
>
> * 使用最小堆优化了每轮查找最近未确定点的过程
> * 入堆过程可能产生冗余数据，处理方法是用continue进行“懒惰删除”，跳过劣质数据
> * 时间复杂度为$O(mlogn)$

## Bellman-Ford

> 单源最短路算法，采用暴力松弛，特点：**第 k 次循环的结果是走 k 条边的最短路**
> 可处理带**负权边**的图，但时间复杂度$O(VE)$较高
> 基于其特点，常用于解决**限制 k 步的最短路**问题，需要backup数组
>
> **模板：**
>
> ```cpp
> int n, m, k;	// n表示结点数，m表示边数，k是路径的最大边数限制
> int dist[N], backup[N];		// dist[x]存储起点到x的最短路距离，backup为其上一轮结果备份
> 
> // 边，u表示出点，v表示入点，w表示边的权重
> struct Edge {
>        int u, v, w;
> } edges[M];
> 
> // 求1到n在k步限制下的最短路距离
> void bellmanFord() {
>        memset(dist, 0x3f, sizeof dist);
>        dist[1] = 0;
> 	
>        // 基于限制，只迭代k次。若无限制，迭代n-1次即可
>        for (int i = 0; i < k; i++) {
>            // 每次松弛前创建并利用上一次结果的备份，防止串联更新
>            memcpy(backup, dist, sizeof dist);
> 
>            // 暴力松弛所有边
>            for (int j = 0; j < m; j++) {
>                int u = edges[j].u, v = edges[j].v, w = edges[j].w;
>                dist[v] = min(dist[v], backup[u] + w);
>            }
>        }
> }
> ```
>
> * 标准Bellman-Ford：没有k步限制，不需要backup，迭代`n-1`次即可
> * backup备份：**作用是锁定"上一轮"的状态，防止本轮更新产生连环影响（串联），保证第 i 次循环只利用第 i-1 次的结果进行扩展。**
>   如果只用一个dist数组（像标准的 BF 写法），在第 i 轮循环中，如果更新了节点 A，紧接着节点 B 又利用更新后的 A 进行了更新，这导致在这一轮里实际上走了两条边（串联效应）。**加了备份数组，就能严格控制层数。**
> * 不可达判定条件：`if (dist[n] > 0x3f3f3f3f / 2) then 不可达`
>   **为什么不是 `== 0x3f3f3f3f`？** 因为 Bellman-Ford 算法处理的是**负权边**。 假设 `n` 号点实际上是不可达的（它是 INF），但有一个点 `u` 也是不可达的（INF），且有一条负权边 `u -> n` 权重为 `-10`。 计算时：`dist[n] = min(INF, INF - 10)`。 虽然数学上 $\infty - 10 = \infty$，但在计算机里 `0x3f3f3f3f - 10` 是一个比 `0x3f3f3f3f` 稍微小一点点的数。 如果不加 `/ 2` 的判断，程序会误以为找到了路。`INF / 2` 是一个经验值，只要比最大可能的路径权值大即可。

## SPFA

> 使用队列优化了Bellman-Ford算法的松弛过程
> 可处理带**负权边**的图，且平均时间复杂度优秀$O(E)$
> 最泛用的单源最短路算法，也可**判断负环**，但遇到网格图会退化为$O(VE)$
> 出于稳定性考虑，图中无负权边时，永远优先使用**堆优化Dijkstra**
> 仅在有负权边的图中使用SPFA
>
> **模板：**
>
> ```cpp
> typedef pair<int, int> PII;
> 
> int n, m;	// 结点数，边数
> // 邻接表：{目标点, 权值}
> vector<vector<PII>> adj(N);
> int dist[N];
> 
> // st[i] 含义的关键区别：
> // Dijkstra: st[i]=true 表示点 i 的最短路"已经彻底确定"，以后不再变。
> // SPFA:     st[i]=true 表示点 i "正在队列中"，防止重复入队。
> bool st[N];
> 
> void addEdge(int u, int v, int w) {
>        adj[u].push_back({v, w});
> }
> 
> int spfa() {
>        memset(dist, 0x3f, sizeof dist);
>        dist[1] = 0;
> 
>        queue<int> q;
>        q.push(1);
>        st[1] = true;
> 
>        while (!q.empty()) {
>            auto t = q.front();
>            q.pop();
>            // 关键点：出队后，立刻取消标记
>            st[t] = false;
> 
>            // 遍历所有邻居
>            for (auto [v, w] : adj[t]) {
>                // 松弛操作：如果通过 t 到 v 距离更短
>                if (dist[t] + w < dist[v]) {
>                    dist[v] = dist[t] + w;
>                    // 优化核心：如果 v 已经在队列里等着被处理了，就不用再加一次
>                    if (!st[v]) {
>                        q.push(v);
>                        st[v] = true;
>                    }
>                }
>            }
>        }
> 
>        return dist[n];
> }
> ```
>
> * 用队列优化更新最短距离的过程，核心思想是：
>   `dist[u]`发生改变，`dist[u] + w`才有可能满足`< dist[u]`
>   用队列保存最短距离发生改变的顶点
>   用`st`记录在队列中的结点，避免重复更新
>
> * 不可达判定条件：`if (dist[n] > 0x3f3f3f3f / 2) then 不可达`
>   
> * 若要判断负环，则需要额外维护一个数组`cnt`，用于记录各个最短路径的边数，当`边数≥顶点数n`时，则一定存在负环（抽屉原理）
>
> **负环判断：**
>
> ```cpp
> typedef pair<int, int> PII;
> 
> int n, m;
> vector<vector<PII>> adj(N);
> int dist[N];
> int cnt[N]; // 新增：记录最短路边数
> bool st[N];
> 
> void addEdge(int u, int v, int w) {
>     adj[u].push_back({v, w});
> }
> 
> // 返回 true 表示存在负环，false 表示不存在
> bool spfa() {
>     // 1. 初始化
>     // 求负环时，通常不需要初始化 dist 为无穷大，
>     // 因为我们关心的是"有没有负权导致无限松弛"，而不是具体的距离。
>     // 当然，为了严谨，初始化为0即可（相当于假设有一个虚拟源点连向所有点，边权为0）。
>     // memset(dist, 0x3f, sizeof dist); 
> 
>     queue<int> q;
> 
>     // 2. 关键点：将所有点入队
>     // 如果只把起点 1 入队，那么只能判断"从 1 出发能到达的负环"。
>     // 如果图中有独立的负环（1走不到），就检测不出来。
>     // 因此，为了判断整张图是否有负环，通常将所有点入队。
>     for (int i = 1; i <= n; i++) {
>         st[i] = true;
>         q.push(i);
>     }
> 
>     while (!q.empty()) {
>         int t = q.front();
>         q.pop();
>         st[t] = false;
> 
>         for (auto [v, w] : adj[t]) {
>             if (dist[t] + w < dist[v]) {
>                 dist[v] = dist[t] + w;
> 
>                 // 3. 维护 cnt 数组
>                 cnt[v] = cnt[t] + 1;
> 
>                 // 4. 判断负环
>                 // 如果边数 >= n，说明经过了 n 条边，也就是 n+1 个点，必然有重复点（成环）
>                 if (cnt[v] >= n) return true; 
> 
>                 if (!st[v]) {
>                     q.push(v);
>                     st[v] = true;
>                 }
>             }
>         }
>     }
> 
>     return false;
> }
> ```
>
> * 若存在负环，则负环上的结点会无限重复松弛过程，直到满足`cnt[v] >= n`，返回true

## Floyd

> 多源汇最短路算法，时间复杂度$O(V^3)$
>
> **模板：**
>
> ```cpp
> // d[i][j] 既是邻接矩阵，最后也是最短距离矩阵
> int d[N][N];
> 
> // 算法结束后，d[a][b]表示a到b的最短距离
> void floyd() {
>     // k 必须在最外层！
>        // 意义：逐步开放中转点。
>        // 第 1 轮：只允许经过 1 号点中转；
>        // ...
>        // 第 k 轮：允许经过 1~k 号点中转。
>        for (int k = 1; k <= n; k ++)
>            for (int i = 1; i <= n; i ++)
>                for (int j = 1; j <= n; j ++)
>                    // 状态转移方程：
>                    // 原路(d[i][j]) vs 经过k中转的路(d[i][k] + d[k][j])
>                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
>    }
> ```
> * 不可达判定条件：`if (d[u][v] > 0x3f3f3f3f / 2) then 不可达`

## Prim

> 最小生成树算法，适合处理**稠密图**，时间复杂度$O(V^2)$
>
> **模板：**
>
> ```cpp
> int g[N][N]; // 邻接矩阵存储图
> int dist[N]; // 【关键差异】存储点到"当前生成树集合"的最小距离
> bool st[N];  // 标记点是否已经加入了生成树点集
> 
> // 返回最小生成树边权之和
> int prim() {
>        // 1. 初始化
>        memset(dist, 0x3f, sizeof dist);
>        dist[1] = 0; // 起点距离集合的距离为0（它自己就是集合的种子）
> 
>        int res = 0; // 存储最小生成树的边权之和
> 
>        // 循环 n 次，每次将一个点加入集合
>        for (int i = 0; i < n; i++) {
>         
>            // 2. 寻找最近点 (跟 Dijkstra 一模一样)
>            // 在所有还没加入生成树的点中，找到距离"集合"最近的点 t
>            int t = -1;
>            for (int j = 1; j <= n; j++)
>                if (!st[j] && (t == -1 || dist[j] < dist[t]))
>                    t = j;
>         
>            // 如果连最近的点都是无穷大，说明图不连通，不存在生成树
>            // 注意：i > 0 是为了排除第一次循环（起点的情况）
>            if (dist[t] == INF) return INF;
>         
>            // 3. 累加权重
>            // 将这条最短边的权值加入结果
>            res += dist[t];
>         
>            // 4. 标记进入集合
>            st[t] = true;
> 
>            // 5. 更新距离 (【关键差异】)
>            // 既然 t 加入了集合，那么其他点到"集合"的距离可能会缩短
>            // 我们看每个点 j：是保持原样近？还是直接连到 t 上更近？
>            for (int j = 1; j <= n; j++)
>                // Dijkstra: min(dist[j], dist[t] + g[t][j]) -> 累加距离
>                // Prim:     min(dist[j], g[t][j])           -> 只看边权
>                dist[j] = min(dist[j], g[t][j]); 
>        }
> 
>        return res;
> }
> ```
>
> * 最小生成树解决的是**无向图**问题，因此邻接矩阵存储边时要添加两条有向边
> * Prim算法和Dijkstra算法非常相似
>   * Prim算法更新**其他点到集合**的距离
>   * Dijkstra算法更新**其它点到起点**的距离
> * 在稀疏图背景下，可类似Dijkstra算法，用堆来优化Prim算法寻找最近点的过程，算法时间复杂度为$O(mlogn)$，但此时性能和kruskal算法接近$O(mlogm)$，而kruskal算法代码更简洁，因此一般不用堆优化的Prim算法

## Kruskal

> 最小生成树算法，适用于**稀疏图**，时间复杂度$O(ElogE)$
>
> **模板：**
>
> ```cpp
> int n, m; // n 个点，m 条边
> 
> // Kruskal 是基于边集的算法，只需要简单地存每一条边一次
> // 结构体存储边：从 u 到 v，权重为 w
> struct Edge {
>        int u, v, w;
> } edges[M];
> 
> // 比较函数：用于 sort 排序
> // 核心贪心思想：权重小的边优先被考虑
> bool cmp(const Edge& a, const Edge& b) {
>        return a.w < b.w;
> }
> 
> // 并查集 p数组和find函数
> int p[N];
> 
> int find(int x) {
>        if (p[x] != x) p[x] = find(p[x]); // 递归路径压缩
>        return p[x];
> }
> 
> int kruskal() {
>        // 1. 初始化并查集：每个点最开始都是独立的集合，父节点是自己
>        for (int i = 1; i <= n; i++)
>            p[i] = i;
> 
>        // 2. 排序：将所有边按权重从小到大排序 O(M log M)
>        sort(edges, edges + m, cmp);
> 
>        int res = 0; // 存储最小生成树的总权重
>        int cnt = 0; // 记录已经加入生成树的边数
> 
>        // 3. 枚举每条边
>        for (int i = 0; i < m; i++) {
>            auto [u, v, w] = edges[i];
> 
>            // 4. 若不形成环，则加入这条边
>            if (find(u) != find(v)) {
>                res += w;     // 将权重加入结果
>                cnt++;        // 边数 +1
>                p[find(u)] = find(v); // 合并集合：将 u 的祖宗指向 v 的祖宗
>            }
>        }
> 
>        // 5. 判断连通性
>        // 生成树必须包含 n-1 条边。如果少于 n-1 条，说明图不连通，返回哨兵值
>        if (cnt < n - 1) return INF;
>        return res;
> }
> ```
>
> * Kruskal是基于边集的算法，只需要简单地用三元组存每一条边一次
> * 并查集的简单应用
> * 时间复杂度$O(mlongm)$，主要来自于排序步骤
> * 需要自定义`std::sort()`的比较器

## 二分图判定与匹配

相关概念[二分图的最大匹配、完美匹配和匈牙利算法](https://www.renfei.org/blog/bipartite-matching.html)

### 染色法

> 用于二分图的判定，时间复杂度$O(V + E)$
> DFS遍历图的过程中，对每个结点进行染色
> 判断原理：
>
> * **一个图是二分图，当且仅当这个图没有奇数环（边数为奇数的环路）**
> * 若存在奇数环，染色时就会发生冲突，说明不是二分图
>
> **模板：**
>
> ```cpp
> int n, m; // n: 点的数量, m: 边的数量
> 
> // color 数组用于记录每个点的颜色
> // 0: 未染色, 1: 颜色A, 2: 颜色B
> int color[N]; 
> 
> vector<vector<int>> adj(N); 
> 
> void addEdge(int u, int v) {
>     adj[u].push_back(v);
> }
> 
> /**
>  * DFS 染色核心逻辑
>  * @param u 当前正在处理的节点
>  * @param c 当前节点 u 需要被染成的颜色 (1 或 2)
>  * @return bool 如果染色过程中没有冲突返回 true，发现冲突返回 false
>  */
> bool dfs(int u, int c) {
>     color[u] = c; // 1. 【标记】先把当前节点染上颜色
> 
>     // 2. 【遍历】遍历 u 的所有邻居
>     for (auto ne : adj[u]) {
>         
>         // 情况 A: 邻居 ne 还没被染过色 (color 为 0)
>         if (!color[ne]) {
>             // 递归去染邻居，颜色应该是 3-c (即 1变2，2变1)
>             // 【关键点】如果递归深层返回 false，说明下面发现了冲突，必须立刻向上层返回 false
>             if (!dfs(ne, 3 - c)) 
>                 return false; 
>         }
>         
>         // 情况 B: 邻居 ne 已经被染过色了，检查是否有冲突
>         // 如果邻居的颜色和当前节点 u 的颜色一样（冲突），说明不是二分图
>         else if (color[ne] == c) 
>             return false; 
>     }
> 
>     // 3. 【成功】遍历完所有邻居都没有发现冲突，说明从 u 开始的这一分枝是合法的
>     return true; 
> }
> 
> int main() {
> 	// 数据输入省略...
> 
>     bool flag = true; // 标志位，默认假设是二分图
>     
>     // 【关键循环】遍历所有点，处理“非连通图”的情况
>     // 图可能由多个不连通的块组成 (森林)，需要确保每个连通块都被检查到
>     for (int i = 1; i <= n; i++) {
>         // 只有当点 i 没被染过色时，才说明它属于一个新的连通块，需要启动一次 DFS
>         if (!color[i]) {
>             // 尝试把该连通块的起点 i 染成颜色 1
>             if (!dfs(i, 1)) {
>                 flag = false; // 只要有一个连通块染色失败，整张图就判否
>                 break;        // 已经失败了，无需继续检查后面的点
>             }
>         }
>     }
> 
>     if (flag) cout << "Yes";
>     else cout << "No";
> 
>     return 0;
> }
> ```
>
> * 二分图染色的经典技巧`3 - c`，实现了在两种颜色之间反复横跳
> * DFS中**失败信号的传递**:
> 	`if (!dfs(...)) return false`，将递归的调用放到`if`判断中，实现了`false`的逐层向上传递
> * main中的for循环遍历所有点：
> 	图不一定是连通的，如果图里有孤立点或者两个分开的圈，只从点 1 开始 DFS 可能会漏掉其他部分。循环检查 `!color[i]` 保证了图的每一个角落都被覆盖到
>
> **应用：**acwing.257

### 匈牙利算法

> 二分图最大匹配算法，时间复杂度$O(VE)$，但实际运行通常小于这个数
> **原理：**
> 通过不停地找增广路来增加匹配中的匹配边和匹配点。找不到增广路时，达到最大匹配（增广路定理）
>
> **模板：**
>
> ```cpp
> int n1, n2, m;      // n1: 左侧点数, n2: 右侧点数, m: 边数
> int match[N];       // match[j] = i 表示：右边的第 j 号点，当前匹配的是左边的第 i 号点
> bool st[N];         // st[j] = true 表示：在当前这一轮模拟中，右边的点 j 已经被预定或询问过了
> vector<vector<int>> adj(N); // 邻接表，存图
> 
> // 建图函数：只存左边指向右边的边即可
> // 因为匈牙利算法只需要从左边出发去寻找右边
> void addEdge(int u, int v) {
>     adj[u].push_back(v);
> }
> 
> // 【核心函数】寻找增广路
> // x: 当前试图找对象的左侧节点（男生）
> bool find(int x) {
>     // 遍历 x 所有的备选目标（他看上的所有女生）
>     for (auto ne : adj[x]) {
>         
>         // 如果这个女生在这一轮查找中还没被问过
>         if (!st[ne]) {
>             st[ne] = true; // 标记一下：这个女生这轮已经被考虑过了，别让别人（或者递归中的自己）再来问了
> 
>             // 【核心逻辑】
>             // 情况 1: match[ne] == 0 -> 这个女生目前单身，直接匹配成功
>             // 情况 2: find(match[ne]) -> 这个女生有男朋友了，但是！
>             //         我们要去问问她的现任男友 (match[ne])：
>             //         "你能不能换一个没被占用的女生？" (递归调用 find)
>             if (match[ne] == 0 || find(match[ne])) {
>                 match[ne] = x; // 协商成功！女生 ne 的对象变成了 x
>                 return true;   // 返回true
>             }
>         }
>     }
> 
>     // 问遍了所有备选，要么都被占了且现任都不肯让位，要么根本没备选，返回false
>     return false; 
> }
> 
> int main() {
> 	// 数据输入省略...
> 
>     int res = 0; // 记录最大匹配数量
> 
>     // 遍历左边的每一个节点（每一个男生），尝试给他们找对象
>     for (int i = 1; i <= n1; i++) {
>         
>         // 【重要】每一轮新的寻找之前，必须清空 st 数组
>         // 含义：为了给 i 找对象，所有的女生在初始状态下都是"未被询问"的
>         // 之前的 st 标记是给上一个男生用的，跟现在无关
>         memset(st, false, sizeof st);
>         
>         // 如果能为 i 找到匹配（或者通过腾挪让出位置）
>         if (find(i)) {
>             res++; // 匹配总数 + 1
>         }
>     }
> 
>     cout << res;
> 
>     return 0;
> }
> ```
>
> **应用：**acwing.372