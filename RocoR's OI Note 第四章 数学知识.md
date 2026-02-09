# RocoR's OI Note 第四章 数学知识

## 数论

### 质数

#### 质数判定——试除法

> 时间复杂度$O(\sqrt{n})$
> ```cpp
> bool isPrime(int x) {
>        if (x < 2) return false;
> 
>        for (int i = 2; i <= x / i; i++)
>            if (x % i == 0) return false;
> 
>        return true;
> }
> ```
>
> * 循环条件：
> 	* 合数的两个因数是成对出现的，枚举时判断较小者即可
> 	* 因此，循环上界为`i <= sqrt(n)`或`i * i <= n`
> 	* 但为提高效率、防止溢出，使用等价形式`i <= n / i`

#### 分解质因数——试除法

> 时间复杂度$O(\sqrt{n})$，通常更快
>
> ```cpp
> // 分解质因数并打印各质因数的底数和指数（个数）
> void divide(int x) {
>        for (int i = 2; i <= x / i; i++) {
>         
>            // 此处如果 i 能整除 x，说明 i 是 x 的一个质因子！！
>            // 原理：因为我们是从小到大枚举，合数因子会被更小的质数提前分解掉
>            if (x % i == 0) {
>                int exp = 0;	// exp 用于记录当前质因子 i 的指数（个数）
>             
>             // 只要还能被 i 整除，就一直除，统计个数
>                while (x % i == 0) {
>                    x /= i;
>                    exp++;
>                }
> 
>                // 输出格式：质因子 指数
>                cout << i << " " << exp << endl;
>            }
>        }
> 
>        // 关键特判：处理剩下的 x
>        // 如果循环结束后 x > 1，说明 x 本身就是一个大于 sqrt(原始x) 的质数
>        // 一个数最多只有一个大于其平方根的质因子
>        if (x > 1) cout << x << " " << 1 << endl;
> }
> ```

#### 筛质数

##### 埃氏筛

> 时间复杂度为$O(nloglogn)$
>
> ```cpp
> int n;					// 筛选上界
> int primes[N], cnt;     // primes[]存储所有质数，cnt记录质数个数，也做写入索引
> bool st[N];             // st[x]存储x是否被筛掉
> 
> void getPrimes(int n) {
>        for (int i = 2; i <= n; i ++ ) {
>            // 如果当前 i 没有被筛过，说明 i 是质数
>            if (!st[i]) primes[cnt++] = i;
>             
>            // 质数的倍数一定不是质数，筛去
>            for (int j = i + i; j <= n; j += i) {
>                st[j] = true;                   
>            }
>        }
> }
> ```
>
> * 核心思想：质数的倍数一定不是质数
> * 算法循环中每次发现的第1个非标记数，一定是质数，加入primes
> * 然而，一个合数可能被筛去多次

##### 线性筛

> 时间复杂度为$O(n)$
>
> ```cpp
> int primes[N], cnt;     // primes[]存储所有素数
> bool st[N];             // st[x]存储x是否被筛掉
> 
> void getPrimes(int n) {
>     for (int i = 2; i <= n; i ++ ) {
>         if (!st[i]) primes[cnt ++ ] = i;
> 
>         // 升序枚举已有的质数 primes[j]
>         for (int j = 0; primes[j] <= n / i; j ++ ) {
>             // 筛掉合数：primes[j] * i
>             // 由于是从小到大枚举，在这里，primes[j] 是这个合数的“最小质因子”
>             st[primes[j] * i] = true;
>             if (i % primes[j] == 0) break;          // 核心逻辑
>         }
>     }
> }
> ```
>
> * 核心思想：每个合数，只被它的“最小质因子”筛掉，且仅筛一次。
> * 对`if (i % primes[j] == 0) break`的理解：
> 	* 如果内层枚举进行到 i 能被 `primes[j]` 整除，说明 i 已经包含了 primes[j] 这个因子
> 	* 这意味着 `primes[j]` 是 i 的最小质因子，或写作 `i = k * primes[j]`
> 	* 如果我们继续循环用 `primes[j+1]` 去乘 i
> 	* 得到的新合数 $X$ = `primes[j+1] * i` = `primes[j+1] * (k * primes[j])` = `primes[j] * (k * primes[j+1])`
> 	* 最小质因子依然是 primes[j]，而非 primes[j + 1]，因为是升序枚举的
> 	* **结论**： 既然 $X$ 的最小质因子是 `primes[j]`，那么 $X$ 应该等到未来，当外层 i 枚举到 `i' = k * primes[j+1]` 的时候，被 `primes[j]` 筛掉。 我们现在如果不 break，就是“越俎代庖”，抢了未来 `primes[j]` 的工作，而且用的是错误的（非最小）质因子

### 约数

#### 求约数——试除法

> 时间复杂度为$O(\sqrt{n})$
>
> ```cpp
> // 返回x的所有约数构成的数组，升序排列
> vector<int> getDivisors(int x) {
>     vector<int> divisors;
>     for (int i = 1; i <= x / i; i++) {
>         if (x % i == 0) {
>             divisors.push_back(i);
>             // 计算并放入较大的约数，注意避免i = sqrt(x)时的两个相同约数重复放入数组
>             if (x / i != i) divisors.push_back(x / i);
>         }
>     }
> 
>     sort(divisors.begin(), divisors.end());
> 
>     return divisors;
> }
> ```
>
> * 约数成对出现，枚举较小者，即可计算较大者
> * int最大值才有大约1500个约数

#### 求约数个数

> **约数个数公式：**
> 设正整数 $N$ 的质因数分解式为：\[ n = p_1^{a_1} \times p_2^{a_2} \times \dots \times p_k^{a_k} \]
>
> 则 $N$ 的约数个数为：\[(a_1 + 1)(a_2 + 1) \dots (a_k + 1)\]
>
> ```cpp
> unordered_map<int, int> mp;		// 用哈希表保存质数的指数
> 
> // 分解质因数
> void divide(int x) {
>     for (int i = 2; i <= x / i; i++) {
>         while (x % i == 0) {
>             x /= i;
>             mp[i]++;
>         }
>     }
>     if (x > 1) mp[x]++;
> }
> 
> // ----------------主函数逻辑----------------
> 
> // 约数个数定理
> LL res = 1;
> for (auto e : mp) res = res * (e.second + 1);
> ```
>
> **约数个数公式推导:**
>
> **核心逻辑：乘法原理 (Combinatorics)**
>
> 任意约数 $d$ 都是由 $N$ 的质因子组合而成的。
>
> 设约数 $d$ 的形式为：
>
> $$d = p_1^{c_1} \times p_2^{c_2} \times \dots \times p_k^{c_k}$$
>
> 对于每个质因子 $p_i$，其指数 $c_i$ 的取值范围是：
>
> $$0 \le c_i \le a_i$$
>
> - **选法分析**：对于质因子 $p_i$，我们可以选 0 个（即 $p_i^0=1$），选 1 个，...，直到选 $a_i$ 个。
> - **独立选择**：因此，每个 $p_i$ 都有 **$(a_i + 1)$** 种选择。
> - **总数计算**：根据乘法原理，不同质因子的选择是独立的，总方案数即为各选择数的乘积。

#### 求约数之和

> **约数之和公式**：
>
> 根据算术基本定理，若整数 $N$ 分解质因数为：
>
> $$N = p_1^{a_1} \times p_2^{a_2} \times \dots \times p_k^{a_k}$$
>
> 那么 $N$ 的所有约数之和 $\sigma(N)$ 为：
>
> $$\sigma(N) = (p_1^0 + p_1^1 + \dots + p_1^{a_1}) \times (p_2^0 + p_2^1 + \dots + p_2^{a_2}) \times \dots \times (p_k^0 + p_k^1 + \dots + p_k^{a_k})$$
>
> ```cpp
> unordered_map<int, int> mp;		// 用哈希表保存质数的指数
> 
> // 分解质因数
> void divide(int x) {
>     for (int i = 2; i <= x / i; i++) {
>         while (x % i == 0) {
>             x /= i;
>             mp[i]++;
>         }
>     }
>     if (x > 1) mp[x]++;
> }
> 
> // ----------------主函数逻辑----------------
> 
> LL res = 1; // 最终结果，初始为 1（乘法单位元）
> 
> // 遍历 map 中每一个质因子及其总指数
> for (const auto &e : mp) {
>     LL p = e.first, a = e.second;	// e.first 是质数 p，e.second 是指数 a
> 
>     // 【核心逻辑】计算单个括号内的等比数列和：(p^0 + p^1 + ... + p^a)
>     LL sum = 1;
>     while (a -- ) sum = (sum * p + 1) % mod;
> 
>     // 将该质因子的结果乘到总结果中
>     res = res * sum % mod;
> }
> ```
>
> * 核心逻辑：迭代求单个括号内的等比数列和：$(p^0 + p^1 + ... + p^a)$
>
> 	* 初始 `sum = 1`
>
> 	* 第 1 次循环：`sum = 1 * p + 1` $\rightarrow (p + 1)$
>
> 	* 第 2 次循环：`sum = (p + 1) * p + 1` $\rightarrow (p^2 + p + 1)$
>
> 	* ...
>
> 	* 第 $a$ 次循环：得到 $(p^a + \dots + p + 1)$

#### 最大公约数

> 欧几里得算法
>
> ```cpp
> // 返回a和b的最大公约数
> int gcd(int a, int b) {
>     return (b == 0) ? a : gcd(b, a % b);
> }
> ```
>
> * $gcd(a,b)=gcd(b,a\ mod\ b)$
> * $gcd(a,0)=a$

## 欧拉函数



## 高斯消元

## 简单博弈论

