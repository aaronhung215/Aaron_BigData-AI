---
layout: post
title: Leetcode Practice - Python
date: 2023-01-15
tags: leetcode, python
categories: leetcode python
comments: true
---

# Leetcode - Array&String
## Easy
### 13. Roman to Integer https://leetcode.com/problems/roman-to-integer/
```python!
class Solution:
    def romanToInt(self, s: str) -> int:
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                  'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        for i in range(len(s)):
            if i < len(s) - 1 and values[s[i]] < values[s[i+1]]:
                ans -= values[s[i]]
            else:
                ans += values[s[i]]
        return ans

```

### 14. Longest Common Prefix
https://leetcode.com/problems/longest-common-prefix/
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        last = '' if not strs else strs.pop()
        for i, c in enumerate(last):
            for s in strs:
                if i >= len(s) or s[i] != c:
                    return last[:i]
        return last

```




### 28. Implement strStr()
https://leetcode.com/problems/implement-strstr/
```python!
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        h, n = len(haystack), len(needle)
        if n == 0: return 0
        if h < n: return -1
        i, next_ = 0, [0] * n
        for j in range(1, n):
            while i > 0 and needle[i] != needle[j]:
                i = next_[i - 1]
            i += needle[i] == needle[j]
            next_[j] = i
        i = 0
        for j in range(h):
            while i > 0 and needle[i] != haystack[j]:
                i = next_[i - 1]
            i += needle[i] == haystack[j]
            if i == n: return j - i + 1
        return -1
```
### 53. Maximum Subarray
https://leetcode.com/problems/maximum-subarray/

> Python 關於正負無窮float('inf')
> max(80, 100, 1000) :  1000

```python!
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_ending_here, max_so_far = float('-inf'), float('-inf')
        for num in nums:
            max_ending_here = max(num, max_ending_here + num)
            max_so_far = max(max_so_far, max_ending_here)
        return max_so_far

```

### 121. Best Time to Buy and Sell Stock
> https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
> map() https://www.runoob.com/python/python-func-map.html
> operator.sub(a, b)
operator.__sub__(a, b)
回傳 a - b。
```python!
import operator
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_ending_here, max_so_far = 0, 0
        for profit in map(operator.sub, prices[1:], prices):
            max_ending_here = max(max_ending_here + profit, 0)
            max_so_far = max(max_so_far, max_ending_here)
        return max_so_far

```

### 1816. Truncate Sentence
https://leetcode.com/problems/truncate-sentence/

```python

class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        def gen(I):
            nonlocal k
            for c in s:
                if c == ' ':
                    k -= 1
                    if k == 0: break
                yield c

        return ''.join(gen(iter(s)))

```