if 1: # Min heap impl
	def _minheappush(heap, item):
		heap.append(item)
		_minsiftup(heap, len(heap)-1)
	def _minheappop(heap):
		last = heap.pop()
		if len(heap) > 0:
			first = heap[0]
			heap[0] = last
			_minsiftdown(heap, 0)
			return first
		return last
	def _minheapify(heap):
		n = len(heap)
		for i in range(1,n):
			_minsiftup(heap, i)
	def _minsiftdown(heap, pos):
		n = len(heap)
		lc = ((pos+1)<<1)-1
		rc = lc+1
		while True:
			# No children
			if lc >= n: break
			# Select child to use for sifting
			prio_child = lc if rc >= n or heap[lc] < heap[rc] else rc
			# Child is larger than parent, swap and keep sifting
			if heap[prio_child] < heap[pos]: heap[prio_child], heap[pos], pos = heap[pos], heap[prio_child], prio_child
			# Child is smaller than parent, we are done
			else: break
			# Update child pointer
			lc = ((pos+1)<<1)-1
			rc = lc+1
	def _minsiftup(heap, pos):
		while pos > 0:
			par = (pos-1)>>1
			if heap[par] > heap[pos]:
				heap[par], heap[pos] = heap[pos], heap[par]
				pos = par
			else: break
	class MinHeap:
		def __init__(self, data=None):
			if data is None: self.data = []
			else:
				self.data = data
				_minheapify(self.data)
		def push(self, item): _minheappush(self.data, item)
		def pop(self): return _minheappop(self.data)
		def remove(self, i):
			n = len(self.data)
			if i < 0 or i >= n: raise IndexError("Index out of bounds")
			ret = self.data[i]
			self.data[i], self.data[n-1] = self.data[n-1], self.data[i]
			self.data.pop()
			if i < n-1:
				_maxsiftdown(self.data, i)
				_maxsiftup(self.data, i)
			return ret
		def index(self, v): return self.data.index(v)
		def clear(self): self.data.clear()
		def __len__(self): return len(self.data)
		def __getitem__(self, i): return self.data[i]
		def __iter__(self): return iter(self.data)
if 1: # Max heap impl
	def _maxheappush(heap, item):
		heap.append(item)
		_maxsiftup(heap, len(heap)-1)
	def _maxheappop(heap):
		last = heap.pop()
		if len(heap) > 0:
			first = heap[0]
			heap[0] = last
			_maxsiftdown(heap, 0)
			return first
		return last
	def _maxheapify(heap):
		n = len(heap)
		for i in range(1,n):
			_maxsiftup(heap, i)
	def _maxsiftdown(heap, pos):
		n = len(heap)
		lc = ((pos+1)<<1)-1
		rc = lc+1
		while True:
			# No children
			if lc >= n: break
			# Select child to use for sifting
			prio_child = lc if rc >= n or heap[lc] > heap[rc] else rc
			# Child is larger than parent, swap and keep sifting
			if heap[prio_child] > heap[pos]: heap[prio_child], heap[pos], pos = heap[pos], heap[prio_child], prio_child
			# Child is smaller than parent, we are done
			else: break
			# Update child pointer
			lc = ((pos+1)<<1)-1
			rc = lc+1
	def _maxsiftup(heap, pos):
		while pos > 0:
			par = (pos-1)>>1
			if heap[par] < heap[pos]:
				heap[par], heap[pos] = heap[pos], heap[par]
				pos = par
			else: break
	class MaxHeap:
		def __init__(self, data=None):
			if data is None: self.data = []
			else:
				self.data = data
				_maxheapify(self.data)
		def push(self, item): _maxheappush(self.data, item)
		def pop(self): return _maxheappop(self.data)
		def remove(self, i):
			n = len(self.data)
			if i < 0 or i >= n: raise IndexError("Index out of bounds")
			ret = self.data[i]
			self.data[i], self.data[n-1] = self.data[n-1], self.data[i]
			self.data.pop()
			if i < n-1:
				_maxsiftdown(self.data, i)
				_maxsiftup(self.data, i)
			return ret
		def index(self, v): return self.data.index(v)
		def clear(self): self.data.clear()
		def __len__(self): return len(self.data)
		def __getitem__(self, i): return self.data[i]
		def __iter__(self): return iter(self.data)
if 1: # Queues based on min and max heap
	class HeapPrioQueue:
		def __init__(self, data=None): raise NotImplementedError("Abstract HeapPrioQueue should not be initialized")
		def push(self, prio, item): self.heap.push((prio, item))
		def push_overflow(self, prio, item, max_size): raise NotImplementedError("Abstract HeapPrioQueue should not be initialized")
		def pop(self): return self.heap.pop()
		def peek(self): return self.heap[0]
		def remove(self, i): return self.heap.remove(i)
		def index(self, v): return next((i for i,x in enumerate(self.heap) if x[1] == v), None)
		def clear(self): self.heap.clear()
		def __len__(self): return len(self.heap)
		def __getitem__(self, i): return self.heap[i]
		def __iter__(self): return iter(self.heap)
		def keys(self): return (v[0] for v in self.heap)
		def values(self): return (v[1] for v in self.heap)
	class MinPrioQueue(HeapPrioQueue):
		def __init__(self, data=None): self.heap = MinHeap(data)
		def push_overflow(self, prio, item, max_size):
			if len(self.heap) == max_size:
				if self.heap[0][0] > prio: return (prio, item)
				ret = self.heap.pop()
			else: ret = None
			self.heap.push((prio, item))
			return ret
	class MaxPrioQueue(HeapPrioQueue):
		def __init__(self, data=None): self.heap = MaxHeap(data)
		def push_overflow(self, prio, item, max_size):
			if len(self.heap) == max_size:
				if self.heap[0][0] < prio: return (prio, item)
				ret = self.heap.pop()
			else: ret = None
			self.heap.push((prio, item))
			return ret

if 1: # Dual heap impl
	def _indexed_swap(heap, idx, inv_idx, i, j):
		if i==j: return
		(
			heap[i],
			heap[j],
			idx[i],
			idx[j],
		) = (
			heap[j],
			heap[i],
			idx[j],
			idx[i],
		)
		(inv_idx[idx[i]],inv_idx[idx[j]]) = (i,j)
	def _indexed_dualheappush(minheap, minidx, maxheap, maxidx, item):
		n = len(minheap)
		minheap.append(item)
		maxheap.append(item)
		minidx.append(n)
		maxidx.append(n)
		_indexed_minsiftup(minheap, minidx, maxidx, n)
		_indexed_maxsiftup(maxheap, maxidx, minidx, n)
	def _indexed_dualheappopmin(minheap, minidx, maxheap, maxidx):
		return _indexed_dualheapminremove(minheap, minidx, maxheap, maxidx, 0)
	def _indexed_dualheappopmax(minheap, minidx, maxheap, maxidx):
		return _indexed_dualheapmaxremove(minheap, minidx, maxheap, maxidx, 0)
	def _indexed_dualheapminremove(minheap, minidx, maxheap, maxidx, minpos):
		n = len(minheap)
		if minpos < 0 or minpos >= n: raise IndexError("Index out of bounds")
		if n == 1: return (minheap.pop(), minidx.pop(), maxheap.pop(), maxidx.pop())[0]
		# Get position of the "to remove" item
		maxpos = minidx[minpos]
		# Swap item to be removed to the end of the heaps
		_indexed_swap(minheap, minidx, maxidx, minpos, n-1)
		_indexed_swap(maxheap, maxidx, minidx, maxpos, n-1)
		# Pop item from all lists and store minheap item
		ret = (minheap.pop(), minidx.pop(), maxheap.pop(), maxidx.pop())[0]
		# Sift down the swapped position in both heaps
		if minpos < n-1:
			_indexed_minsiftdown(minheap, minidx, maxidx, minpos)
			_indexed_minsiftup(minheap, minidx, maxidx, minpos)
		if maxpos < n-1:
			_indexed_maxsiftdown(maxheap, maxidx, minidx, maxpos)
			_indexed_maxsiftup(maxheap, maxidx, minidx, maxpos)
		# Return result
		return ret
	def _indexed_dualheapmaxremove(minheap, minidx, maxheap, maxidx, maxpos):
		n = len(minheap)
		if maxpos < 0 or maxpos >= n: raise IndexError("Index out of bounds")
		if n == 1: return (minheap.pop(), minidx.pop(), maxheap.pop(), maxidx.pop())[0]
		# Get position of the "to remove" item
		minpos = maxidx[maxpos]
		# Swap item to be removed to the end of the heaps
		_indexed_swap(minheap, minidx, maxidx, minpos, n-1)
		_indexed_swap(maxheap, maxidx, minidx, maxpos, n-1)
		# Pop item from all lists and store maxheap item
		ret = (minheap.pop(), minidx.pop(), maxheap.pop(), maxidx.pop())[2]
		# Sift down the swapped position in both heaps
		if minpos < n-1:
			_indexed_minsiftdown(minheap, minidx, maxidx, minpos)
			_indexed_minsiftup(minheap, minidx, maxidx, minpos)
		if maxpos < n-1:
			_indexed_maxsiftdown(maxheap, maxidx, minidx, maxpos)
			_indexed_maxsiftup(maxheap, maxidx, minidx, maxpos)
		# Return result
		return ret
	def _indexed_dualheapify(minheap, minidx, maxheap, maxidx):
		_indexed_minheapify(minheap, minidx, maxidx)
		_indexed_maxheapify(maxheap, maxidx, minidx)

	def _indexed_minheapify(heap, idx, inv_idx):
		n = len(heap)
		for i in range(1,n):
			_indexed_minsiftup(heap, idx, inv_idx, i)
	def _indexed_minsiftdown(heap, idx, inv_idx, pos):
		n = len(heap)
		lc = ((pos+1)<<1)-1
		rc = lc+1
		while True:
			# No children
			if lc >= n: break
			# Select child to use for sifting
			prio_child = lc if rc >= n or heap[lc] < heap[rc] else rc
			# Child is larger than parent, swap and keep sifting
			if heap[prio_child] < heap[pos]:
				_indexed_swap(heap, idx, inv_idx, prio_child, pos)
				pos = prio_child
			# Child is smaller than parent, we are done
			else: break
			# Update child pointer
			lc = ((pos+1)<<1)-1
			rc = lc+1
	def _indexed_minsiftup(heap, idx, inv_idx, pos):
		while pos > 0:
			par = (pos-1)>>1
			if heap[par] > heap[pos]:
				_indexed_swap(heap, idx, inv_idx, par, pos)
				pos = par
			else: break

	def _indexed_maxheapify(heap, idx, inv_idx):
		n = len(heap)
		for i in range(1,n):
			_indexed_maxsiftup(heap, idx, inv_idx, i)
	def _indexed_maxsiftdown(heap, idx, inv_idx, pos):
		n = len(heap)
		lc = ((pos+1)<<1)-1
		rc = lc+1
		while True:
			# No children
			if lc >= n: break
			# Select child to use for sifting
			prio_child = lc if rc >= n or heap[lc] > heap[rc] else rc
			# Child is larger than parent, swap and keep sifting
			if heap[prio_child] > heap[pos]:
				_indexed_swap(heap, idx, inv_idx, prio_child, pos)
				pos = prio_child
			# Child is smaller than parent, we are done
			else: break
			# Update child pointer
			lc = ((pos+1)<<1)-1
			rc = lc+1
	def _indexed_maxsiftup(heap, idx, inv_idx, pos):
		while pos > 0:
			par = (pos-1)>>1
			if heap[par] < heap[pos]:
				_indexed_swap(heap, idx, inv_idx, par, pos)
				pos = par
			else: break

	class DualHeap:
		def __init__(self, data=None):
			if data is None:
				self.minheap, self.maxheap = [], []
				self.minidx, self.maxidx = [], []
			else:
				self.minheap, self.maxheap = list(data), list(data)
				self.minidx, self.maxidx = list(range(len(data))), list(range(len(data)))
				_indexed_minheapify(self.minheap, self.minidx, self.maxidx)
				_indexed_maxheapify(self.maxheap, self.maxidx, self.minidx)
		def push(self, item): _indexed_dualheappush(self.minheap, self.minidx, self.maxheap, self.maxidx, item)
		def popmin(self): return _indexed_dualheappopmin(self.minheap, self.minidx, self.maxheap, self.maxidx)
		def popmax(self): return _indexed_dualheappopmax(self.minheap, self.minidx, self.maxheap, self.maxidx)
		def peekmin(self): return self.minheap[0]
		def peekmax(self): return self.maxheap[0]
		def remove(self, i): return self.removemin(i)
		def removemin(self, i): return _indexed_dualheapminremove(self.minheap, self.minidx, self.maxheap, self.maxidx, i)
		def removemax(self, i): return _indexed_dualheapmaxremove(self.minheap, self.minidx, self.maxheap, self.maxidx, i)
		def index(self, v): return self.minheap.index(v)
		def clear(self):
			self.minheap.clear()
			self.minidx.clear()
			self.maxheap.clear()
			self.maxidx.clear()
		def indexmin(self, v): return self.minheap.index(v)
		def indexmax(self, v): return self.maxheap.index(v)
		def __len__(self): return len(self.minheap)
		def __getitem__(self, i): return self.minheap[i]
		def getmin(self, i): return self.minheap[i]
		def getmax(self, i): return self.minheap[i]
		def __iter__(self): return iter(self.minheap)
		def itermin(self): return iter(self.minheap)
		def itermax(self): return iter(self.maxheap)
if 1: # Queues based on capped dual heap
	class DualPrioQueue(HeapPrioQueue):
		def __init__(self, data=None): self.heap = DualHeap(data)
		def push_overflow(self, prio, item, max_size): self.push_overflow_min(self, prio, item, max_size)
		def push_overflow_max(self, prio, item, max_size):
			if len(self.heap) == max_size:
				if self.heap.peekmax()[0] < prio: return (prio, item)
				ret = self.heap.popmax()
			else: ret = None
			self.heap.push((prio, item))
			return ret
		def push_overflow_min(self, prio, item, max_size):
			if len(self.heap) == max_size:
				if self.heap.peekmin()[0] > prio: return (prio, item)
				ret = self.heap.popmin()
			else: ret = None
			self.heap.push((prio, item))
			return ret
		def pop(self): return self.heap.popmin()
		def popmin(self): return self.heap.popmin()
		def popmax(self): return self.heap.popmax()
		def remove(self, i): return self.heap.remove(i)
		def removemin(self, i): return self.heap.removemin(i)
		def removemax(self, i): return self.heap.removemax(i)
		def index(self, v): return next((i for i,x in enumerate(self.heap.itermin()) if x[1] == v), None)
		def indexmin(self, v): return next((i for i,x in enumerate(self.heap.itermin()) if x[1] == v), None)
		def indexmax(self, v): return next((i for i,x in enumerate(self.heap.itermax()) if x[1] == v), None)
		def peekmin(self): return self.heap.peekmin()
		def peekmax(self): return self.heap.peekmax()
		def getmin(self, i): return self.heap.getmin(i)
		def getmax(self, i): return self.heap.getmax(i)
		def itermin(self): return self.heap.itermin()
		def itermax(self): return self.heap.itermax()
		def keysmin(self): return (v[0] for v in self.heap.itermin())
		def keysmax(self): return (v[0] for v in self.heap.itermax())
		def valuesmin(self): return (v[1] for v in self.heap.itermin())
		def valuesmax(self): return (v[1] for v in self.heap.itermax())

if 1: # Limited size list
	class CappedList:
		def __init__(self, capacity, data=None):
			if data is None: data = []
			if len(data) > capacity: raise IndexError("Data exceeds capacity")
			self.capacity = capacity
			self.len = len(data)
			self.data = [None if i >= len(data) else data[i] for i in range(capacity)]
		def append(self, item):
			if self.len == self.capacity: raise IndexError("List is full")
			self.data[self.len], self.len = item, self.len+1
		def pop(self, index=None):
			if index is None: index = self.len-1
			if index < 0 or index >= self.len: raise IndexError("Index out of bounds")
			ret = self.data[index]
			for i in range(index+1, self.len): self.data[i-1] = self.data[i]
			self.data[self.len-1], self.len = None, self.len-1
			return ret
		def capacity(self): return self.capacity
		def index(self, v): return self.data[:self.len].index(v)
		def clear(self): self.data.clear()
		def __len__(self): return self.len
		def __getitem__(self, i):
			if i >= self.len or i < 0: raise IndexError("Index out of bounds")
			return self.data[i]
		def __setitem__(self, i, v):
			if i >= self.len or i < 0: raise IndexError("Index out of bounds")
			self.data[i] = v
		def __iter__(self): return iter(self.data[:self.len])
	class CappedMinHeap(MinHeap):
		def __init__(self, capacity, data=None):
			if data is None: self.data = CappedList(capacity)
			else:
				self.data = CappedList(capacity, data)
				_minheapify(self.data)
		def capacity(self): return self.data.capacity
	class CappedMaxHeap(MaxHeap):
		def __init__(self, capacity, data=None):
			if data is None: self.data = CappedList(capacity)
			else:
				self.data = CappedList(capacity, data)
				_maxheapify(self.data)
		def capacity(self): return self.data.capacity
	class CappedDualHeap(DualHeap):
		def __init__(self, capacity, data=None):
			if data is None:
				self.minheap, self.maxheap = CappedList(capacity), CappedList(capacity)
				self.minidx, self.maxidx = CappedList(capacity), CappedList(capacity)
			else:
				self.minheap, self.maxheap = CappedList(capacity, data), CappedList(capacity, data)
				self.minidx, self.maxidx = CappedList(capacity, list(range(len(data)))), CappedList(capacity, list(range(len(data))))
				_indexed_dualheapify(self.minheap, self.minidx, self.maxheap, self.maxidx)
		def capacity(self): return self.minheap.capacity
	class CappedMinPrioQueue(MinPrioQueue):
		def __init__(self, capacity, data=None): self.heap = CappedMinHeap(capacity, data)
		def capacity(self): return self.heap.capacity()
	class CappedMaxPrioQueue(MaxPrioQueue):
		def __init__(self, capacity, data=None): self.heap = CappedMaxHeap(capacity, data)
		def capacity(self): return self.heap.capacity()
	class CappedDualPrioQueue(DualPrioQueue):
		def __init__(self, capacity, data=None): self.heap = CappedDualHeap(capacity, data)
		def push_overflow(self, prio, item): raise NotImplementedError("Use push_overflow_min or push_overflow_max instead")
		def push_overflow_max(self, prio, item):
			if len(self.heap) == self.heap.capacity():
				if self.heap.peekmax()[0] < prio: return (prio, item)
				ret = self.heap.popmax()
			else: ret = None
			self.heap.push((prio, item))
			return ret
		def push_overflow_min(self, prio, item):
			if len(self.heap) == self.heap.capacity():
				if self.heap.peekmin()[0] > prio: return (prio, item)
				ret = self.heap.popmin()
			else: ret = None
			self.heap.push((prio, item))
			return ret
		def capacity(self): return self.heap.capacity()

if __name__ == "__main__":
	import numpy as np
	rnd = np.random.sample(1000)
	# Sort with minheap
	minheap = MinHeap()
	for v in rnd: minheap.push(v)
	minheap_sorted = np.array([minheap.pop() for _ in range(rnd.shape[0])])
	# Sort with maxheap
	maxheap = MaxHeap()
	for v in rnd: maxheap.push(v)
	maxheap_sorted = np.array([maxheap.pop() for _ in range(rnd.shape[0])])
	# Sort with dual heap ascending
	dualheap = DualHeap()
	for i,v in enumerate(rnd):
		dualheap.push(v)
		assert dualheap.peekmin() == np.min(rnd[:i+1])
		assert dualheap.peekmax() == np.max(rnd[:i+1])
	dualheap_sorted1 = np.array([dualheap.popmin() for _ in range(rnd.shape[0])])
	# Sort with dual heap descending
	dualheap = DualHeap()
	for i,v in enumerate(rnd):
		dualheap.push(v)
		assert dualheap.peekmin() == np.min(rnd[:i+1])
		assert dualheap.peekmax() == np.max(rnd[:i+1])
	dualheap_sorted2 = np.array([dualheap.popmax() for _ in range(rnd.shape[0])])
	# Sort with dual heap alternating
	dualheap = DualHeap()
	for i,v in enumerate(rnd):
		dualheap.push(v)
		assert dualheap.peekmin() == np.min(rnd[:i+1])
		assert dualheap.peekmax() == np.max(rnd[:i+1])
	dualheap_sorted3 = rnd.copy()
	i,j = 0,len(rnd)-1
	while i < j:
		assert dualheap.peekmin() == min(v for v in dualheap), (dualheap.peekmin(), min(v for v in dualheap))
		assert dualheap.peekmax() == max(v for v in dualheap), (dualheap.peekmax(), max(v for v in dualheap))
		dualheap_sorted3[i] = dualheap.popmin()
		dualheap_sorted3[j] = dualheap.popmax()
		i,j = i+1,j-1
	# Sort with numpy (min to max)
	np_sorted = np.sort(rnd)
	# Test
	assert np.all(minheap_sorted == np_sorted)
	assert np.all(maxheap_sorted == np_sorted[::-1])
	assert np.all(dualheap_sorted1 == np_sorted)
	assert np.all(dualheap_sorted2 == np_sorted[::-1])
	assert np.all(dualheap_sorted3 == np_sorted)

	# Test CappedDualPrioQueue
	cdpq = CappedDualPrioQueue(10)
	for i,v in enumerate(rnd):
		if len(cdpq) > 0:
			assert cdpq.peekmin()[0] == min(cdpq.keys())
			assert cdpq.peekmax()[0] == max(cdpq.keys())
			assert cdpq.peekmin()[0] == list(cdpq.keysmin())[0]
			assert cdpq.peekmax()[0] == list(cdpq.keysmax())[0]
			curr_max = cdpq.peekmax()[0]
		assert np.all(np.sort(list(cdpq.keysmin())) == np.sort(list(cdpq.keysmax())))
		popped = cdpq.push_overflow_max(v, i)
		if popped is not None:
			assert popped[0] >= curr_max
			assert popped[0] >= max(cdpq.keys())
		assert cdpq.peekmin()[0] == min(cdpq.keys())
		assert cdpq.peekmax()[0] == max(cdpq.keys())
		assert cdpq.peekmin()[0] == list(cdpq.keysmin())[0]
		assert cdpq.peekmax()[0] == list(cdpq.keysmax())[0]
		assert cdpq.peekmin()[0] == np.min(rnd[:i+1])
		assert cdpq.peekmax()[0] == np.sort(rnd[:i+1])[len(cdpq)-1]
	cdpq_k_min = np.array([cdpq.popmin()[0] for _ in range(len(cdpq))])
