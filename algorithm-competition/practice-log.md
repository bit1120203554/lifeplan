# 刷题记录

> 记录每道题的学习情况。格式：
> - `[日期] 题号/来源 · 标签 · 状态: AC/WA/TLE/学习 · 备注`

## Week 1 (6/28 - 7/4)

<!-- 在此记录 -->

## Week 2 (7/5 - 7/11)

<!-- 在此记录 -->

## Week 3 (7/12 - 7/18)

<!-- 在此记录 -->

## Week 4 (7/19 - 7/25)

<!-- 在此记录 -->

## Week 5 (7/26 - 8/1)

<!-- 在此记录 -->

---

## 错题集

| 题号 | 错误原因 | 正确做法 | 复习标记 |
|------|----------|----------|----------|
| | | | |

---

## 常用模板

记录你常用的算法模板，方便比赛时快速复制：

### BFS
```cpp
queue<int> q;
vector<bool> vis(N, false);
q.push(start);
vis[start] = true;
while (!q.empty()) {
    int u = q.front(); q.pop();
    for (int v : adj[u]) {
        if (!vis[v]) {
            vis[v] = true;
            q.push(v);
        }
    }
}
```

### 二分查找（左闭右开）
```cpp
int l = 0, r = n;
while (l < r) {
    int mid = (l + r) / 2;
    if (check(mid)) r = mid;
    else l = mid + 1;
}
// l 是第一个满足 check 的位置
```

### 0-1背包
```cpp
vector<int> dp(W+1, 0);
for (int i = 0; i < n; i++)
    for (int w = W; w >= weight[i]; w--)
        dp[w] = max(dp[w], dp[w - weight[i]] + value[i]);
```
