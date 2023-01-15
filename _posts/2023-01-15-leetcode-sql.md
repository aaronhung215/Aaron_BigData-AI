---
layout: post
title: Leetcode Practice - SQL
date: 2023-01-15
tags: leetcode, sql
categories: leetcode sql
comments: true
---

Leetcode sql practice

# Leetcode - SQL
1. https://leetcode.com/problems/second-highest-salary/
## > 176. Second Highest Salary
```sql
WITH CTE AS
(SELECT Salary, DENSE_RANK () OVER (ORDER BY Salary desc) AS RANK_desc
FROM Employee)
SELECT MAX(salary) AS SecondHighestSalary
FROM CTE
WHERE RANK_desc = 2
```


2. https://leetcode.com/problems/nth-highest-salary/
## > 177. Nth Highest Salary

```sql=
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
    
    WITH CTE AS
    (SELECT Salary, DENSE_RANK () OVER (ORDER BY Salary desc) AS RANK_desc
    FROM Employee)
    SELECT MAX(salary) AS SecondHighestSalary
    FROM CTE
    WHERE RANK_desc = N
      
  );
END


```

3. https://leetcode.com/problems/consecutive-numbers/submissions/ 
**Difficult**
## > 180. Consecutive Numbers
> Write an SQL query to find all numbers that appear at least three times consecutively. 


```sql=
# Write your MySQL query statement below
select distinct Num as ConsecutiveNums
from Logs
where (Id + 1, Num) in (select * from Logs) and (Id + 2, Num) in (select * from Logs)

```

4. https://leetcode.com/problems/department-highest-salary/

## > 184. Department Highest Salary



```sql
# Write your MySQL query statement below
select c.name as Department, a.name as Employee, a.salary as Salary
from Employee a,
(select departmentId, MAX(salary) salary
from Employee
group by departmentId) b,
Department c
where a.salary = b.salary and a.departmentId = b.departmentId
and a.departmentId = c.id


```

5. https://leetcode.com/problems/department-top-three-salaries/submissions/
**Difficult**
## > 185. Department Top Three Salaries

```sql
SELECT Department, Employee, Salary
from
(
SELECT 
    b.name as Department,
    a.name Employee,
    Salary,
    DENSE_RANK() OVER ( PARTITION BY departmentID ORDER BY salary DESC) as rank
FROM Employee a
LEFT JOIN Department b ON a.departmentID = b.id 
)
where rank <=3

```

6. https://leetcode.com/problems/rank-scores/submissions/

**Difficult**

## > 178. Rank Scores

> The scores should be ranked from the highest to the lowest.
> If there is a tie between two scores, both should have the same ranking.
> After a tie, the next ranking number should be the next consecutive
> integer value. In other words, there should be no holes between ranks.
> Return the result table ordered by score in descending order.
```sql!

select 
score,
dense_rank() OVER (order by score desc) as rank
from Scores



SELECT
  Score,
  (SELECT count(distinct Score) FROM Scores WHERE Score >= s.Score) Rank
FROM Scores s
ORDER BY Score desc
-- 固定 s.score, 再比較內圈的score全部資料
```
![](https://i.imgur.com/IYT7UHL.png)


## 判斷非空值
```sql
 寻找用户推荐人
🚀 给定表 customer ，里面保存了所有客户信息和他们的推荐人。
+------+------+-----------+
| id   | name | referee_id|
+------+------+-----------+
|    1 | Will |      NULL |
|    2 | Jane |      NULL |
|    3 | Alex |         2 |
|    4 | Bill |      NULL |
|    5 | Zack |         1 |
|    6 | Mark |         2 |
+------+------+-----------+
🚀 需求
写一个查询语句，返回一个客户列表，列表中客户的推荐人的编号都不是2。
对于上面的示例数据，结果为：
+------+
| name |
+------+
| Will |
| Jane |
| Bill |
| Zack |
+------+
🐴🐴 答案
# Write your MySQL query statement below
select name  from customer
where IFNULL(referee_id,0) <> 2
--mysql判断非空的函数
ISNULL(expr)    如果expr为null返回值1，否则返回值为0
IFNULL(expr1,expr2) 如果expr1值为null返回expr2的值，否则返回expr1的值
/* Write your T-SQL query statement below */
select name  from customer
where referee_id <> 2 OR referee_id IS NULL
/* Write your PL/SQL query statement below */
select name "name"  from customer
where nvl(referee_id,0) <> 2

```

## 1873. Calculate Special Bonus 判斷奇數和字首
https://hackmd.io/@SupportCoding/r1eZnN5yj
```sql!
🚀 表: Employees
+-------------+---------+
| 列名        | 类型     |
+-------------+---------+
| employee_id | int     |
| name        | varchar |
| salary      | int     |
+-------------+---------+
employee_id 是这个表的主键。
此表的每一行给出了雇员id ，名字和薪水。
 
🚀 需求
写出一个SQL 查询语句，计算每个雇员的奖金。如果一个雇员的id是奇数并且他的名字不是以'M'开头，那么他的奖金是他工资的100%，否则奖金为0。
Return the result table ordered by employee_id.
返回的结果集请按照employee_id排序。
查询结果格式如下面的例子所示。

 示例 1:

输入：
Employees 表:
+-------------+---------+--------+
| employee_id | name    | salary |
+-------------+---------+--------+
| 2           | Meir    | 3000   |
| 3           | Michael | 3800   |
| 7           | Addilyn | 7400   |
| 8           | Juan    | 6100   |
| 9           | Kannon  | 7700   |
+-------------+---------+--------+
输出：
+-------------+-------+
| employee_id | bonus |
+-------------+-------+
| 2           | 0     |
| 3           | 0     |
| 7           | 7400  |
| 8           | 0     |
| 9           | 7700  |
+-------------+-------+
解释：
因为雇员id是偶数，所以雇员id 是2和8的两个雇员得到的奖金是0。
雇员id为3的因为他的名字以'M'开头，所以，奖金是0。
其他的雇员得到了百分之百的奖金。


🐴🐴 答案
# Write your MySQL query statement below
select 
employee_id,
case when mod(employee_id,2)=1 and LEFT(name,1)!='M' then salary
else 0 end bonus
from Employees
order by employee_id

/* Write your T-SQL query statement below */
select 
employee_id,
case when employee_id%2=1 and SUBSTRING(name,1,1)!='M' then salary
else 0 end bonus
from Employees
order by employee_id


/* Write your PL/SQL query statement below */
select 
employee_id "employee_id",
case when mod(employee_id,2)=1 and substr(name,1,1)!='M' then salary
else 0 end  "bonus"
from Employees
order by 1

```

## (M)534. Game Play Analysis III 

```sql=
SELECT a.player_id, a.event_date,
 (SELECT SUM(games_played) FROM Activity AS b
  WHERE a.player_id = b.player_id
  AND a.event_date >= b.event_date) AS games_played_so_far
FROM Activity AS a



SELECT a1.player_id, a1.event_date, SUM(a2.games_played) AS games_so_far
  FROM activity a1
  JOIN activity a2     ON a1.player_id = a2.player_id
                      AND a1.event_date >=a2.event_date
 GROUP BY a1.player_id, a1.event_date
 ORDER BY a1.player_id, a1.event_date


```


## (Medium) 550. Game Play Analysis IV
https://zhuanlan.zhihu.com/p/254592333
```sql=
SELECT ROUND(COUNT(DISTINCT b.player_id)/COUNT(DISTINCT a.player_id), 2) AS fraction FROM Activity AS a
LEFT JOIN
(SELECT player_id, MIN(event_date) AS first_login FROM Activity
GROUP BY player_id) AS b
ON a.player_id = b.player_id
AND DATEDIFF(a.event_date, b.first_login) = 1

SELECT ROUND(COUNT(DISTINCT b.player_id)/COUNT(DISTINCT a.player_id), 2) AS fraction FROM Activity AS a
LEFT JOIN
(SELECT player_id, FIRST_VALUE(event_date) OVER(PARTITION BY player_id ORDER BY event_date) AS first_login FROM Activity) AS b
ON a.player_id = b.player_id
AND DATEDIFF(a.event_date, b.first_login) = 1
```



## (Hard)569. Median Employee Salary


```sql
SELECT id, company, salary
*,
COUNT() OVER (PARTITION BY company) cnt
ROW_NUMBER OVER (PARTITION BY company ORDER BY salary) rnum
FROM employee e
WHERE (cnt/2) <= rnum AND rnum<= (cnt/2) + 1
ORDER BY company, salary

```


## (Medium) 262. Trips and Users
https://zhuanlan.zhihu.com/p/252454836
```sql
SELECT a.Request_at AS Day,
1-ROUND(SUM(CASE WHEN a.Status = 'completed' THEN 1 ELSE 0 END)/COUNT(*),2) AS 'Cancellation Rate' FROM Trips AS a
INNER JOIN 
(SELECT * FROM Users
 WHERE Role = 'client') AS b
ON a.Client_Id = b.Users_Id
INNER JOIN 
(SELECT * FROM Users
 WHERE Role = 'driver') AS c
ON a.Driver_Id = c.Users_Id
WHERE b.Banned = 'No'
AND c.Banned = 'No'
AND a.Request_at BETWEEN '2013-10-01' AND '2013-10-03'
GROUP BY a.Request_at
ORDER BY a.Request_at

```


## (Hard) 571. Find Median Given Frequency of Numbers
https://zhuanlan.zhihu.com/p/257945802

```sql 

SELECT AVG(Number) AS median
FROM 
(
select *, SUM(Frequency) OVER (ORDER BY Number) AS cum_sum, 
          (SUM(Frequency) OVER ())/2.0  AS mid
FROM Numbers
) AS temp
WHERE mid BETWEEN cum_sum - frequency AND cum_sum;

```

```sql

SELECT DISTINCT  B.TB010, MB002,
SUM(B.TB019) over(PARTITION BY B.TB010) Qty,    --計算數量
SUM(B.TB019) over(ORDER BY B.TB010) Acc_Qty ,   --累積數量
SUM(B.TB019) over() Total_Qty                   --總數量
FROM POSTB AS  A
INNER JOIN  POSTB AS B ON A.TB001= B.TB001 AND
A.TB002= B.TB002 AND
A.TB003= B.TB003 AND
A.TB006= B.TB006
INNER JOIN  INVMB ON MB001 = B.TB010
WHERE A.TB010 = '5000001' AND B.TB010  '5000001' --AND B.TB001 ='20180731'
ORDER BY Qty desc

```


## (Medium) 574. Winning Candidate
https://zhuanlan.zhihu.com/p/258311949

```sql
SELECT Name FROM Candidate AS a
JOIN
(SELECT CandidateId FROM Vote
 GROUP BY CandidateId
 ORDER BY COUNT(*) DESC
 LIMIT 1
) AS b
ON a.id = b.CandidateId;

```


## (Hard) 579. Find Cumulative Salary of an Employee
https://blog.csdn.net/chelseajcole/article/details/81532054

```sql

select id, month, Salary
FROM
(
select 
    id,
    month,
    SUM(salary) OVER(PARTITION BY ID Order by month ROWS 2 PRECEDING ) as Salary,
    DENSE_RANK() OVER(PARTITION BY id order by month desc)month_no
)src
where month_no > 1
ORDER BY id, month desc


```

## (Hard) 601. Human Traffic of Stadium
https://medium.com/data-science-for-kindergarten/leetcode-mysql-601-human-traffic-of-stadium-9bbfb30240de

```sql

# Write your MySQL query statement below
SELECT distinct tb1.id as id, tb1.visit_date, tb1.people
FROM Stadium as tb1, Stadium as tb2, Stadium as tb3 
WHERE (tb1.people >= 100 and tb2.people >= 100 and tb3.people >= 100) and
      (
      (tb1.id+1=tb2.id and tb1.id+2=tb3.id) or
      (tb1.id-1=tb2.id and tb1.id+1=tb3.id) or
       tb1.id-1=tb2.id and tb1.id-2=tb3.id
      )
ORDER BY id

```