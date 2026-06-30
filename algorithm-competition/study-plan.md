# 码蹄杯备考计划（C++）

> 从零竞赛经验到能做出题。时间：5周（2026.06.28 - 2026.08.02）
> 语言：C++（有C++基础，需熟悉STL竞赛用法）

---

## 总体策略

**核心原则**: 不做"系统学习"，而是"以题带学"。
你这5周不可能系统学完所有算法竞赛知识。正确做法是：
1. 先掌握竞赛C++的必备套路（STL、输入输出加速）
2. 直接刷真题和模拟题
3. 遇到不会的算法再学

**目标定位**: 比赛中做出签到题（最简单的一两道）+ 部分中等题。不追求AC所有题。

**优势**: 已经有C++基础，可以跳过语言学习阶段，直接进入STL和算法。

---

## 📚 必须掌握的知识点（按优先级）

### 第一优先级：保底必学（1-2周）
这些是几乎所有比赛都会考的基础，学了就能拿分：

1. **C++竞赛输入输出**
   - `ios::sync_with_stdio(false); cin.tie(0);` 加速
   - `getline(cin, s)` / `while (cin >> x)` 多组输入
   - `printf/scanf` 格式化（保留小数、补零等）
   - 文件重定向（有的比赛需要）

2. **STL容器 —— 竞赛必会！**
   - **vector**: 动态数组，`push_back`, `resize`, `sort`
   - **stack**: 栈，`push`, `pop`, `top`（括号匹配、单调栈）
   - **queue**: 队列，`push`, `pop`, `front`（BFS）
   - **deque**: 双端队列，`push_front/back`, `pop_front/back`
   - **priority_queue**: 优先队列（堆），默认最大堆（Dijkstra、Top-K）
   - **set / multiset**: 有序集合，`insert`, `erase`, `find`, `lower_bound`
   - **map / unordered_map**: 键值对映射，竞赛常用哈希版
   - **pair / tuple**: 存储一对/多元组
   - **string**: `find`, `substr`, `+` 拼接，`to_string`/`stoi`

3. **STL算法**
   - `sort(v.begin(), v.end())` + 自定义比较器
   - `reverse`, `unique`（去重前先排序）
   - `lower_bound` / `upper_bound`（二分查找，必须在有序数组上）
   - `next_permutation` / `prev_permutation`（全排列枚举）
   - `max_element` / `min_element` / `accumulate`
   - `fill` / `memset`（初始化数组）
   - `__builtin_popcount`（位运算计数）

4. **暴力/模拟**
   - 按题意一步步模拟
   - DFS穷举所有可能
   - 位运算枚举子集

### 第二优先级：核心算法（2-3周）
这些是中等题的常见考点：

5. **动态规划（DP）基础**
   - 背包问题（0-1背包、完全背包）← **高频考点**
   - 线性DP（最长上升子序列 LIS、最长公共子序列 LCS）
   - 区间DP（矩阵连乘、石子合并）
   - 状态压缩DP（小规模，n ≤ 20 左右）

6. **图论基础**
   - DFS / BFS —— **最常考，务必掌握**
   - 最短路径：Dijkstra（堆优化版，用 `priority_queue`）
   - 并查集（Union-Find）：路径压缩 + 按秩合并
   - 拓扑排序（入度法）
   - 最小生成树：Kruskal（并查集实现）

7. **数学基础**
   - 最大公约数/最小公倍数：`__gcd` / `std::gcd`（C++17）
   - 质数判断 + 埃式筛/欧拉筛
   - 快速幂取模
   - 组合数计算（预处理阶乘和逆元）

8. **字符串处理**
   - KMP 字符串匹配
   - 哈希（字符串哈希判等）

### 第三优先级：进阶（有余力再学）
9. 线段树 / 树状数组（区间查询、单点更新）
10. 最近公共祖先 (LCA) / 树链剖分
11. 数论进阶（逆元、扩展欧几里得、中国剩余定理）
12. 计算几何（点线关系、凸包）

---

## 📅 5周时间分配

### Week 1 (6/28 - 7/4): STL与基础
- 重点：C++ 竞赛输入输出 + STL容器全面掌握
- vector, stack, queue, deque, priority_queue, set, map 写法熟记
- sort, lower_bound, next_permutation 等算法
- 找 2-3 道简单模拟题练手，确保STL用得流畅
- **本周目标**: 看到一道题能用STL把暴力/模拟写出来

### Week 2 (7/5 - 7/11): 基础算法
- DFS 和 BFS —— **重点！几乎必考**
- 二分查找应用
- 简单DP练3-5题（斐波那契、背包、LIS）
- 并查集模板 + 1-2道题
- **本周目标**: DFS/BFS 打得熟练，背包DP能写

### Week 3 (7/12 - 7/18): 核心算法
- 图论：堆优化Dijkstra + 拓扑排序
- 数学：筛质数、快速幂
- **找码蹄杯往年真题做**（这个非常关键，了解题风和难度）
- **本周目标**: 刷完能找到的所有码蹄杯真题

### Week 4 (7/19 - 7/25): 刷题周
- 每日做 2-3 道洛谷 "普及+" 难度的题
- 限时训练（模拟比赛环境）
- 查漏补缺，哪个算法不熟就补哪个
- **本周目标**: 积累手感，保持稳定的做题状态

### Week 5 (7/26 - 8/1): 冲刺
- 每日一套模拟题（Codeforces Div.3 / AtCoder ABC）
- 只看错题和模板回顾
- 准备比赛物资（环境、IDE 配置、模板代码整理）
- **本周目标**: 保持状态，不要学新东西了

---

## 🔍 找题渠道

| 平台 | 用途 | 说明 |
|------|------|------|
| **码蹄杯官网** | 往年真题（最重要！） | 先找到往届题目，最贴近真实难度 |
| **洛谷** | 题库按标签分类 | 搜"普及-"到"普及/提高-"难度 |
| **Codeforces** | 周赛Div.3/Div.4 | 适合练手，难度分层好 |
| **AtCoder Beginner Contest** | A/B/C题 | 新手友好，有官方题解 |
| **LeetCode** | 偏面试 | 可用但和竞赛风格不同 |

### 建议刷题路径
1. **第一步**: 找到码蹄杯往届题目，全部做一遍（最贴近真实难度）
2. **第二步**: 洛谷按标签刷 — 搜"普及-"到"普及/提高-"难度
3. **第三步**: 有余力再做 Codeforces Div.3 或 AtCoder ABC 的A/B/C

---

## ⚔️ 比赛策略

比赛时要做的几件事：

1. **先看所有题** — 花5分钟把所有题目扫一遍，找签到题（最简单的那道）
2. **先做签到题** — 确保AC至少一题保底，稳住心态
3. **再做熟悉的题型** — 看到DP题、BFS题等自己练过的，优先做
4. **不会的题暴力尝试** — 小数据用暴力枚举，有时候能过部分点
5. **不要死磕** — 卡住30分钟换题
6. **注意边界** — 数据范围（int vs long long）、数组大小、输入输出格式

---

## 📝 C++竞赛小抄

```cpp
// ===== 输入输出加速 =====
ios::sync_with_stdio(false);
cin.tie(0);

// ===== 常用类型 =====
using ll = long long;
using pii = pair<int, int>;
#define fi first
#define se second

// ===== vector =====
vector<int> v;
v.push_back(x);
sort(v.begin(), v.end());
sort(v.begin(), v.end(), greater<int>());           // 降序
sort(v.begin(), v.end(), [](int a, int b){return a>b;});

// ===== stack / queue / deque =====
stack<int> st; st.push(x); st.pop(); st.top();
queue<int> q; q.push(x); q.pop(); q.front(); q.back();
deque<int> dq; dq.push_front(x); dq.pop_back();

// ===== priority_queue =====
priority_queue<int> pq;                              // 最大堆
priority_queue<int, vector<int>, greater<int>> pq;   // 最小堆

// ===== set / map =====
set<int> s; s.insert(x); s.count(x); s.erase(x);
map<string, int> mp; mp["key"] = value;
// lower_bound 返回 >= x 的第一个迭代器
auto it = s.lower_bound(x);

// ===== 二分查找 =====
int pos = lower_bound(v.begin(), v.end(), x) - v.begin();  // 第一个 >= x
int pos = upper_bound(v.begin(), v.end(), x) - v.begin();  // 第一个 > x

// ===== 万能头 =====
#include <bits/stdc++.h>
using namespace std;

// ===== 位运算 =====
__builtin_popcount(x);    // 二进制中1的个数
__builtin_clz(x);         // 前导零个数
(x >> k) & 1;             // 取第k位
x & (-x);                 // lowbit

// ===== 快速幂 =====
ll qpow(ll a, ll b, ll mod) {
    ll res = 1;
    while (b) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

// ===== 并查集 =====
int fa[N];
int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
void unite(int x, int y) {
    x = find(x); y = find(y);
    if (x != y) fa[x] = y;
}

// ===== Dijkstra =====
vector<pii> g[N];
int dist[N];
void dijkstra(int s) {
    memset(dist, 0x3f, sizeof(dist));
    dist[s] = 0;
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : g[u]) {
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
}
```
