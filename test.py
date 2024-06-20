from enum import Enum
import numpy as np

class Node:
	def __init__(self, val):
		self.val = val
		self.next = []
		self.prev = []
  
	def add_next(self, n):
		self.next.append(n)
    
	def add_prev(self, p):
		self.prev.append(p)
  
a = Node(1)
b = Node(2)
c = Node(3)

d = [a, b, c]

a.add_prev(b)
c.add_prev(b)
b.add_next(a)
b.add_next(c)

del b
# for n in b.next:
# 	n.prev.remove(b)
 
print(a.next)
print(c.next)
print(d)


e = [('a', 1), ('a', 2)]
f = dict(e)
print(f)

class TTT(Enum):
	AAA = 1
	BBB = 2
	CCC = 3
 
for i in TTT:
	print(i)
	print(i.value)
 
combin_list = []
def find_combination(cnt:int, total:int, combin:list):
	if cnt >= total:
		combin_list.append(combin)
		return
	for i in range(3):
		tmp = combin.copy()
		tmp.append(i)
		find_combination(cnt+1, total, tmp)

find_combination(0, 1, [])

print(combin_list)