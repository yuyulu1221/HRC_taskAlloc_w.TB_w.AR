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