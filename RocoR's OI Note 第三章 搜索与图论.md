# RocoR's OI Note 第三章 搜索与图论

本笔记基于AcWing《算法基础课》第3章 搜索与图论

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
>     e[idx] = b;
>     ne[idx] = adj[a];
>     adj[a] = idx++;
> }
> 
> // 枚举结点 t 的所有邻接点（从 t 能走到的所有结点）
> for (int i = adj[t]; i != -1; i = ne[i]){
>     //......
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
>     adj[a].push_back(b);
> }
> 
> // 枚举结点 t 的所有邻接点（从 t 能走到的所有结点）
> for (auto neighbor : adj[t]){
>     //......
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
>     // 1. 截止条件（递归出口）
>     if (到达终点 || 状态不合法) {
>         更新答案;
>         return;
>     }
> 
>     // 2. 遍历所有可能的“下一步”
>     for (每一个可能的选择 in current_options) {
>         
>         // --- 做出选择 ---
>         标记当前选择 (visited[next] = true);
>         记录路径 (path.push(next));
> 
>         // 3. 下探（进入下一层递归）
>         dfs(下个状态参数);
> 
>         // 4. 回溯（恢复现场 - 非常关键！）
>         // 只有在需要“多种组合”或“多次使用同一资源”时才需要这一步
>         撤销标记 (visited[next] = false);
>         撤销路径 (path.pop());
>         // --- 撤销选择 ---
>     }
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
>    	q.pop();
> 
>     // 4. 枚举 t 能走到的所有下一步 next
>     	for (所有 next from t) {
>         	if (next 合法 && dist[next] == -1) {
>             	dist[next] = dist[t] + 1;
>             	q.push(next);
>         	}
>     	}
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
>     // 1. 初始化：遍历所有节点（注意题目通常是1到n），将初始入度为0的点入队
>     for (int i = 1; i <= n; i++)
>         if (d[i] == 0)
>             q[++tt] = i; // 入队操作：队尾指针后移并存入值
> 
>     // 2. BFS 循环处理
>     while (hh <= tt){
>         // 出队操作：取出队头元素，队头指针后移
>         int t = q[hh++];
> 
>         // 遍历当前节点 t 的所有邻居 u (即存在边 t -> u)
>         for (auto neighbor : adj[t]){
>             int u = neighbor;
> 
>             // 核心逻辑：既然 t 已经处理完（被删去），u 的入度减 1
>             d[u]--;
> 
>             // 如果 u 的入度变为 0，说明 u 的所有前置条件都满足了，可以入队
>             if (d[u] == 0) q[++tt] = u;
>         }
>     }
> 
>     // 3. 判断结果
>     // 队列中存储了所有进入拓扑序列的点。
>     // 如果入队的点总数等于 n ，说明成功排好序。
>     // 如果小于 n，说明图中存在环，导致环上的点入度永远无法减为 0，也就无法入队。
>     return tt == n - 1;
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
> memset(dist, 0x3f, sizeof dist);        // 0x3f3f3f3f作为距离的“最大值”
> dist[1] = 0;                            // 起点到自己的距离为0
> 
> for (int i = 0; i < n - 1; i ++ )       // 执行n-1次（自己到自己的距离已经确定）
> {
>   // 在还未确定最短路的点中，寻找距离最小的点
>   int t = -1;
>   for (int j = 1; j <= n; j ++ )
>       if (!st[j] && (t == -1 || dist[t] > dist[j]))
>           t = j;
>     
>   // 标记该点最短路已确定
>   st[t] = true;
> 
>   // 用t更新其他点的距离（松弛）
>   for (int j = 1; j <= n; j ++ )
>       dist[j] = min(dist[j], dist[t] + g[t][j]);
> }
> 
> if (dist[n] == 0x3f3f3f3f) return -1;       // 不可达
> return dist[n];
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
>     adj[x].push_back({y, w});
> }
> 
> int dijkstra() {
>     memset(dist, 0x3f, sizeof dist);
>     // 定义优先队列（小根堆）
>     // 堆中存储格式：{距离, 节点编号},因为pair默认按first排序
>     priority_queue<PII, vector<PII>, greater<PII>> heap;
>     dist[1] = 0;
>     heap.push({dist[1], 1});
> 
>     while(!heap.empty()) {
>         // 结构化绑定，取出堆顶（当前距离源点最近的未确定点）
>         auto [d, u] = heap.top(); // d是当前点距离，u是当前点编号
>         heap.pop();
>     
>         if (st[u]) continue;
>         // 标记该点最短路已确定
>         st[u] = true;
>         
> 		// 扫描t的所有邻居进行松弛操作
>         for (auto [v, w] : adj[u]) { // v是目标点编号，w是u->v这条边权值
>             if (dist[v] > d + w) {
>                 dist[v] = d + w;
>                 heap.push({dist[v], v}); // 将新产生的路径结点入堆
>         }
>     }
> }
> 
>     if (dist[n] == 0x3f3f3f3f) return -1;
>     return dist[n];
> }
> ```
>
> * 使用最小堆优化了每轮查找最近未确定点的过程
> * 入堆过程可能产生冗余数据，处理方法是用continue进行“懒惰删除”，跳过劣质数据
> * 时间复杂度为$O(mlogn)$

## Bellman-Ford

> 算法采用暴力松弛，特点：**第 k 次循环的结果是走 k 条边的最短路**
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
>     int u, v, w;
> } edges[M];
> 
> // 求1到n在k步限制下的最短路距离
> void bellmanFord() {
>     memset(dist, 0x3f, sizeof dist);
>     dist[1] = 0;
> 	
>     // 基于限制，只迭代k次。若无限制，迭代n-1次即可
>     for (int i = 0; i < k; i++) {
>         // 每次松弛前创建并利用上一次结果的备份，防止串联更新
>         memcpy(backup, dist, sizeof dist);
> 
>         // 暴力松弛所有边
>         for (int j = 0; j < m; j++) {
>             int u = edges[j].u, v = edges[j].v, w = edges[j].w;
>             dist[v] = min(dist[v], backup[u] + w);
>         }
>     }
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
> int n, m;
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
> adj[u].push_back({v, w});
> }
> 
> int spfa() {
> memset(dist, 0x3f, sizeof dist);
> dist[1] = 0;
> 
> queue<int> q;
> q.push(1);
> st[1] = true;
> 
> while (!q.empty()) {
> auto t = q.front();
> q.pop();
> // 关键点：出队后，立刻取消标记
> st[t] = false;
> 
> // 遍历所有邻居
> for (auto [v, w] : adj[t]) {
>    // 松弛操作：如果通过 t 到 v 距离更短
>    if (dist[t] + w < dist[v]) {
>        dist[v] = dist[t] + w;
>        // 优化核心：如果 v 已经在队列里等着被处理了，就不用再加一次
>        if (!st[v]) {
>            q.push(v);
>            st[v] = true;
>        }
>    }
> }
> }
> 
> return dist[n];
> }
> ```
>
> * 用队列优化更新最短距离的过程，核心思想是：
>   `dist[u]`发生改变，`dist[u] + w`才有可能满足`< dist[u]`
>   用队列保存最短距离发生改变的顶点
>   用st记录在队列中的结点，避免重复更新
>
> * 若要判断负环，则需要额外维护一个数组cnt，用于记录各个最短路径的边数，当边数≥顶点数n时，则一定存在负环（抽屉原理）
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
> * 若存在负环，则会无限重复松弛过程，直到满足`cnt[v] >= n`，返回true