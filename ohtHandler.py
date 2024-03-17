import therbligHandler as tbh

TBH = tbh.TBHandler()
OHT = TBH.get_OHT().list

class OHTNode(object):
	def __init__(self, val):
		self.val = val
		self.next = []
	
	def insert(self, val, ):
		if self.next.count() == 0:
			self.next.append(OHTNode(val))
		else:
			self.next[0].insert(val)

