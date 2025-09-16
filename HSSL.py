import numpy as np
import heapq
import matplotlib.pyplot as plt
# import copy
import graphidxbaselines as gib
import plotly.graph_objects as go
import h5py
import time
from scipy.cluster.hierarchy import DisjointSet
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from local_py.VPTree import VPTree
from local_py.heaps import MinPrioQueue
from tqdm.auto import tqdm
from more_itertools import peekable
# import torch
# assert torch.cuda.is_available()
# print(f"Using GPU '{torch.cuda.get_device_name(0)}'")

#%% ##### Line drawing #####
def min_distance_to_line_segment(x1, y1, x2, y2, x, y):
    # Calculate the length of the line segment
    line_length = ((x2 - x1)**2 + (y2 - y1)**2)**.5
    if line_length == 0: return ((x - x1)**2 + (y - y1)**2)**.5
    # Calculate the projection of the point onto the line
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length**2
    if t < 0: return ((x - x1)**2 + (y - y1)**2)**.5
    elif t > 1: return ((x - x2)**2 + (y - y2)**2)**.5
    else: # Closest point is within the line segment
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        return ((x - closest_x)**2 + (y - closest_y)**2)**.5
def draw_line(canvas, x1, y1, x2, y2, color=0, width=1, alpha=1):
    if abs(x1-x2) < abs(y1-y2): # More vertical than horizontal
        if y1 > y2: x1, y1, x2, y2 = x2, y2, x1, y1
        xstart = int(x1 - width/2)
        ystart = int(y1 - width/2)
        xend = int(x2 + width/2) + 1
        yend = int(y2 + width/2) + 1
        nxhalf = int(width/2)+1
        xfun = lambda y: (x2-x1)/(y2-y1)*(y-y1) + x1
        for y in range(ystart, yend+1):
            if y < 0 or y >= canvas.shape[1]: continue
            if y2-y1 == 0: continue
            xlow = int(xfun(y) - nxhalf)
            for dx in range(2*nxhalf+1):
                x = xlow + dx
                if x < 0 or x >= canvas.shape[0]: continue
                dist = min_distance_to_line_segment(x1, y1, x2, y2, x, y)
                if dist < width/2+.5:
                    factor = max(0,min(min(1, width), 1 - (dist+.5-width/2)))
                    canvas[x, y] = canvas[x, y]*(1-alpha*factor) + color*alpha*factor
    else: # More horizontal than vertical
        if x1 > x2: x1, y1, x2, y2 = x2, y2, x1, y1
        xstart = int(x1 - width/2)
        ystart = int(y1 - width/2)
        xend = int(x2 + width/2) + 1
        yend = int(y2 + width/2) + 1
        nyhalf = (int(width/2)+1)
        yfun = lambda x: (y2-y1)/(x2-x1)*(x-x1) + y1
        for x in range(xstart, xend+1):
            if x < 0 or x >= canvas.shape[0]: continue
            if x2-x1 == 0: continue
            # print(f"yfun and x: {x, x1, x2, y1, y2}")
            ylow = int(yfun(x) - nyhalf)
            for dy in range(2*nyhalf+1):
                y = ylow + dy
                if y < 0 or y >= canvas.shape[1]: continue
                dist = min_distance_to_line_segment(x1, y1, x2, y2, x, y)
                if dist < width/2+.5:
                    factor = max(0, min(min(1, width), 1 - (dist+.5-width/2)))
                    canvas[x, y] = canvas[x, y]*(1-alpha*factor) + color*alpha*factor

# %% ##### Dendrogram sorting and plotting #####
def roundup_fix_dendrogram(dendrogram):
    N = len(dendrogram)+1
    for i in range(len(dendrogram)):
        dendrogram[i] = list(dendrogram[i])
        if dendrogram[i][0] >= N: dendrogram[i][2] = max(dendrogram[i][2], dendrogram[dendrogram[i][0]-N][2])
        if dendrogram[i][1] >= N: dendrogram[i][2] = max(dendrogram[i][2], dendrogram[dendrogram[i][1]-N][2])
def elki_sort_dendrogram(dendrogram):
    check_monotone = lambda: all(a[2] < b[2] for a,b in zip(dendrogram[:-1], dendrogram[1:]))
    if check_monotone(): return dendrogram
    N = len(dendrogram)+1
    # m = len(dendrogram), n = N
    # Sort by max height, merge height, merge number
    order1 = np.array([v[2] for v in sorted([(size, distance, i) for i, (_, _, distance, size) in enumerate(dendrogram)])])
    # Now we need to ensure merges are consistent in their order
    seen = np.zeros(len(dendrogram), dtype=bool)
    order2 = np.zeros(len(dendrogram), dtype=int)
    size = 0
    def add_recursive(size, i):
        left, right = dendrogram[i][0]-N, dendrogram[i][1]-N
        # a = left, b = right
        for child in [left, right]:
            if child >= 0 and not seen[child]:
                size = add_recursive(size, child)
                size += 1
                order2[size] = child
                seen[child] = True
        return size
    for i in order1:
        if seen[i]: continue
        size = add_recursive(size, i)
        size += 1
        order2[size] = i
        seen[i] = True
    assert size == len(dendrogram)
    reverse_order = np.zeros(len(dendrogram), dtype=int)
    for i,j in enumerate(order2): reverse_order[j] = i
    return [
        [
            order2[dendrogram[i_old][0]],
            order2[dendrogram[i_old][1]],
            dendrogram[i_old][2],
            dendrogram[i_old][3],
        ]
        for i_old in reverse_order
    ]
def plotly_dendrogram(dendrogram, prerender=True, min_size=1, line_width=1, width=1000, height=400, largest_left=True):
    N = dendrogram[-1][3]
    lines = []
    work = [[0, N-1, *dendrogram[-1], True]]
    xlabels = [None for _ in range(N)]
    while len(work) > 0:
        min_x, max_x, left_id, right_id, distance, size, large_left = work.pop()
        assert size == N or size > min_size
        left_size = dendrogram[left_id-N][3] if left_id >= N else 1
        right_size = dendrogram[right_id-N][3] if right_id >= N else 1
        left_distance = dendrogram[left_id-N][2] if left_size > min_size else 0
        right_distance = dendrogram[right_id-N][2] if right_size > min_size else 0
        if (left_size < right_size and large_left) or (left_size > right_size and not large_left):
            left_id, left_size, left_distance, right_id, right_size, right_distance = right_id, right_size, right_distance, left_id, left_size, left_distance
        left_min = min_x
        left_max = min_x + left_size - 1
        right_min = left_max + 1
        right_max = max_x
        left_mid = (left_min+left_max)/2
        right_mid = (right_min+right_max)/2
        lines += [
            [left_mid, left_distance],
            [left_mid, distance],
            [right_mid, distance],
            [right_mid, right_distance],
            [np.nan, np.nan],
        ]
        if left_size <= min_size: xlabels[int(left_mid)] = left_id
        else: work.append([left_min, left_max, *dendrogram[left_id-N], largest_left])
        if right_size <= min_size: xlabels[int(right_mid)] = right_id
        else: work.append([right_min, right_max, *dendrogram[right_id-N], True])
    lines = np.array(lines)
    if 0:
        fig = go.Figure(go.Scatter(
            x = lines[:,0],
            y = lines[:,1],
            mode = "lines",
        ), layout = dict(
            xaxis_tickvals = np.arange(N),
            xaxis_ticktext = xlabels,
            paper_bgcolor="#fff",
            plot_bgcolor="#fff",
            margin={k:0 for k in "tblr"},
            width=width,
            height=height,
        ))
        if not prerender: fig.show(renderer="browser")
        else:
            from IPython.display import Image, display
            fig.update_layout(xaxis_tickvals=[])
            img_bytes = fig.to_image(format="png")
            display(Image(img_bytes))
    else:
        image = np.ones((width, height))
        max_y = np.nanmax(lines[:,1])
        margin = 5
        alpha = .25
        log_y = False
        x_to_screen = lambda x: margin + (width-2*margin) * x / (N-1)
        if log_y:
            min_y = np.nanmin(lines[:, 1][lines[:,1] > 0])
            lminy = np.log(min_y)
            lmaxy = np.log(max_y)
            y_to_screen = lambda y: margin + (height-2*margin) * ( np.log(max(min_y, y)) - lminy ) / ( lmaxy - lminy )
        else:
            y_to_screen = lambda y: margin + (height-2*margin) * y / max_y
        for i in tqdm(range(len(lines)-1), desc="Drawing lines"):
            a, b = lines[i:i+2]
            ax, ay, bx, by = [*a, *b]
            if np.any(np.isnan([*a, *b])): continue
            # print(f"before: {ay}, {by}, {ax}, {bx}")
            ax = x_to_screen(ax)
            bx = x_to_screen(bx)
            ay = y_to_screen(ay)
            by = y_to_screen(by)
            # print(f"after: {ay}, {by}, {ax}, {bx}")
            if ax == bx and ay == by: continue
            draw_line(image, ax, ay, bx, by, width=line_width, alpha=alpha)
        from PIL import Image
        from IPython.display import display
        display(Image.fromarray((255*image.T[::-1]).astype('uint8'), 'L'))
def prune_to_first_npts(dendrogram, npts):
    N = len(dendrogram)+1
    new_dendrogram = []
    # Remap points to a proper cluster index or -1
    index_remap = np.arange(2*N)
    index_remap[npts:N] = -1
    for i_in, (left_child, right_child, distance, size) in enumerate(dendrogram):
        left_child, right_child = int(left_child), int(right_child)
        # Use the remapping
        left_child = index_remap[left_child]
        right_child = index_remap[right_child]
        if left_child < 0 and right_child < 0: index_remap[i_in+N] = -1
        elif left_child < 0: index_remap[i_in+N] = right_child
        elif right_child < 0: index_remap[i_in+N] = left_child
        else:
            index_remap[i_in+N] = npts + len(new_dendrogram)
            left_size = 1 if left_child < npts else new_dendrogram[left_child-npts][3]
            right_size = 1 if right_child < npts else new_dendrogram[right_child-npts][3]
            new_size = left_size + right_size
            new_dendrogram.append([left_child, right_child, distance, new_size])
    assert len(new_dendrogram) == npts-1, f"Dendrogram is too short. It should be {npts-1} but is {len(new_dendrogram)}."
    assert new_dendrogram[-1][3] == npts, f"Last merge should have size {npts} but is {new_dendrogram[-1][3]}."
    return new_dendrogram

# %% ##### Searchers #####
class PrioritySearcher:
	def __init__(self, Tree, q):
		self.tree = Tree
		self.query = q
		self.expand_queue = MinPrioQueue()
	def lower_bound(self):
		return np.inf if len(self.expand_queue) == 0 else self.expand_queue.peek()[0]

	def advance_gen(self):
		self.expand_queue.clear()
		self.expand_queue.push(0, self.tree)
		while len(self.expand_queue) > 0:
			lower_bound, curr_node = self.expand_queue.pop()
			# If the node is a leaf
			if curr_node.is_leaf:
				ids = curr_node.ids
				for id in ids: yield id
			# If the node is not a leaf
			else:
				# Calculate distance to vantage point
				vp_dist = np.linalg.norm(curr_node.vp - self.query)
				lower_bound_inside = max(lower_bound, vp_dist - curr_node.upper_bound_inside)
				lower_bound_outside = max(lower_bound, curr_node.lower_bound_outside - vp_dist)
				
				self.expand_queue.push(lower_bound_inside, curr_node.left)
				self.expand_queue.push(lower_bound_outside, curr_node.right)
class PrioritySearcher_HNSW:
	def __init__(self, data, graph, union_find, q, ef=0):
		self.data = data
		self.graph = graph
		self.query = q
		self.expand_queue = MinPrioQueue()
		self.candidate_queue = MinPrioQueue()
		self.visited = set()
		self.union_find = union_find
		self.ef = ef
	def batch_dists(self, js):
		return np.linalg.norm(self.data[js] - self.data[self.query], axis=1)
	def advance_gen(self, return_self=False):
		if return_self: yield (0, self.query)
		self.expand_queue.clear()
		self.expand_queue.push(0, self.query)
		self.candidate_queue.clear()
		self.visited.clear()
		self.visited.add(self.query)

		while len(self.expand_queue) > 0:
			_, expansion_point = self.expand_queue.pop()
			
			neighbors = self.graph.get_neighbors(layer=0, node=expansion_point)
			neighbors = [n for n in neighbors if n not in self.visited]
			
			# Do not compute distance to the same cluster (also halts navigation here!)
			reduced_neighbors = []
			for n in neighbors:
				if not self.union_find.connected(self.query, n): reduced_neighbors.append(n)
				else: self.visited.add(n)
			neighbors = reduced_neighbors

			# TODO: In a compiler language, do not compute all distances but compute them one-by-one and hope to trigger a merge that makes the other distance computations redundant
			if 1: # Python version
				neighbor_distances = self.batch_dists(neighbors)
				for distance, neighbor in zip(neighbor_distances, neighbors):
					self.visited.add(neighbor)
					self.expand_queue.push(distance, neighbor)
					self.candidate_queue.push(distance, neighbor)
				while len(self.candidate_queue) > self.ef:
					yield self.candidate_queue.pop()
			else: # Compiler version
				for neighbor in neighbors:
					self.visited.add(neighbor)
					neighbor_distance = np.linalg.norm(self.data[self.query] - self.data[neighbor])
					self.expand_queue.push(distance, neighbor)
					yield self.candidate_queue.push_overflow(neighbor_distance, neighbor, self.ef)
		while len(self.candidate_queue) > 0:
			yield self.candidate_queue.pop()

# %% ##### HSSL-VPTrees Turbo #####
def HSSL_Turbo(data, n_trees=1, cuda=False, clean_fraction=2, **vp_kwargs):
    if cuda:
        raise ValueError("Cuda currently not supported")
        data_tensor = torch.Tensor(data).to("cuda:0")
        def batch_dists(i, js):
            return torch.cdist(
                data_tensor[js],
                data_tensor[data[i:i+1]],
            ).cpu().numpy().flatten()
    else:
        def batch_dists(i, js):
            return np.linalg.norm(data[js] - data[i], axis=1)
    M = []
    N = len(data)
    P = []
    H = [[] for _ in range(N)]

    dendrogram = []

    vp_kwargs["store_data"] = False
    trees = [VPTree(data, **vp_kwargs) for _ in range(n_trees)]

    U = DisjointSet(np.arange(len(data)))
    cluster_id_map = np.arange(N)
    cluster_id_counter = N
    cluster_uncleaned_num = np.zeros(N)
    cleaning_cycles = 0

    for i in tqdm(range(N), desc="Initializing"):
        searcher = PrioritySearcher(trees[np.random.randint(n_trees)], data[i])
        P.append((searcher, searcher.advance_gen()))
        
        if 0: # Old version
            while len(H[i]) == 0 or H[i][0][0] > P[i][0].lower_bound():
                j = P[i][1].__next__()
                if j != i:
                    heapq.heappush(H[i], (np.linalg.norm(data[i] - data[j]), j))
        else: # New version
            while len(H[i]) == 0 or H[i][0][0] > P[i][0].lower_bound():
                old_lower_bound = P[i][0].lower_bound()
                js = []
                while P[i][0].lower_bound() == old_lower_bound:
                    j = P[i][1].__next__()
                    if j != i: js.append(j)
                for j, jdist in zip(js, batch_dists(i,js)):
                    heapq.heappush(H[i], (jdist, j))
        
        heapq.heappush(M, (H[i][0][0], i))

    with tqdm(total=N-1, desc="Merging clusters") as pbar:

        while U.n_subsets != 1:
            d, i = heapq.heappop(M)
            if len(H[i]) > 0:
                # print(d, i, "if")
                d2, j = heapq.heappop(H[i])
                if d == d2:
                    if not U.connected(i, j):
                        if U.n_subsets > 2: # Maybe remove all elements from the heaps within the same cluster
                            size_i = U.subset_size(i)
                            size_j = U.subset_size(j)
                            uncleaned_total_i = cluster_uncleaned_num[i] + size_j
                            uncleaned_total_j = cluster_uncleaned_num[j] + size_i
                            new_uncleaned_total = 0
                            cluster_reps = [U[i], U[j]]
                            for v, uncleaned_total, size in [[i, uncleaned_total_i, size_i], [j, uncleaned_total_j, size_j]]:
                                if uncleaned_total / size > clean_fraction:
                                    for k in U.subset(v):
                                        offset = 0
                                        length = len(H[k])
                                        while offset < length:
                                            if U[H[k][offset][1]] in cluster_reps:
                                                # Swap element in the same cluster to the back
                                                H[k][offset], H[k][length-1] = H[k][length-1], H[k][offset]
                                                # Remove it
                                                H[k].pop()
                                                length -= 1
                                            else:
                                                offset += 1
                                        heapq.heapify(H[k])
                                    cluster_uncleaned_num[U[v]] = 0
                                    cleaning_cycles += 1
                                    pbar.desc = f"Merging clusters ({cleaning_cycles})"
                                    uncleaned_total = 0
                                new_uncleaned_total = max(new_uncleaned_total, uncleaned_total)

                        # add (d, i, j) to the dendrogram
                        dendrogram.append((
                            cluster_id_map[U[i]],
                            cluster_id_map[U[j]],
                            d,
                            U.subset_size(i) + U.subset_size(j),
                        ))
                        U.merge(i, j)
                        cluster_id_map[U[i]] = cluster_id_counter
                        cluster_id_counter += 1
                        cluster_uncleaned_num[U[i]] = new_uncleaned_total
                        pbar.update(1)
                        pbar.refresh()
                        if U.n_subsets == 1: break

            if 0: # Old version
                while len(H[i])==0 or H[i][0][0] > P[i][0].lower_bound():
                    try: j = P[i][1].__next__()
                    except StopIteration: break
                    if not U.connected(j, i):
                        heapq.heappush(H[i], (np.linalg.norm(data[i] - data[j]), j))
                    # P[i].advance()
            else: # New version
                # print(d, i, "else")
                rep_i = U[i]
                while len(H[i])==0 or H[i][0][0] > P[i][0].lower_bound():
                    old_lower_bound = P[i][0].lower_bound()
                    if np.isinf(old_lower_bound): break
                    js = []
                    while P[i][0].lower_bound() == old_lower_bound:
                        try: j = P[i][1].__next__()
                        except StopIteration: break
                        if j != i and U[j] != rep_i: js.append(j)
                    for j, jdist in zip(js, batch_dists(i,js)):
                        heapq.heappush(H[i], (jdist, j))
        
            if len(H[i]) > 0: heapq.heappush(M, (H[i][0][0], i))
    assert len(dendrogram) == N-1
    return dendrogram
##### HNSW HSSL #####
def HNSW_HSSL(data, ef=20, **hnsw_kwargs):
    
    M = []
    N = len(data)
    P = []
    
    start_time = time.time()
    total_merges = N - 1
    milestones = {
        int(total_merges * 0.25): None,
        int(total_merges * 0.5): None,
        int(total_merges * 0.75): None,
        int(total_merges * 0.8): None,
        int(total_merges * 0.9): None,
        int(total_merges * 0.95): None,
        int(total_merges * 0.99): None,
    }

    dendrogram = []

    graphs = gib.PyHNSW(data, **hnsw_kwargs)

    U = DisjointSet(np.arange(len(data)))
    cluster_id_map = np.arange(N)
    cluster_id_counter = N

    for i in tqdm(range(N), desc="Initializing"):
        searcher = PrioritySearcher_HNSW(data, graphs, U, i, ef=ef)
        P.append((searcher, peekable(searcher.advance_gen())))
        
        dist, _ = P[i][1].peek()
        heapq.heappush(M, (dist, i))


    with tqdm(total=N-1,desc="Merging clusters") as pbar:

        while U.n_subsets != 1:
            
            # if len(M) == 0:
            #     print(f"{N-1-len(dendrogram)} merges missing") #, returning intermediate result")
            #     # return dendrogram
            
            d, i = heapq.heappop(M)
            try:
                d2, j = P[i][1].__next__()
                if d == d2:
                    if not U.connected(i, j):

                        # add (d, i, j) to the dendrogram
                        dendrogram.append((
                            cluster_id_map[U[i]],
                            cluster_id_map[U[j]],
                            d,
                            U.subset_size(i) + U.subset_size(j),
                        ))
                        U.merge(i, j)
                        cluster_id_map[U[i]] = cluster_id_counter
                        cluster_id_counter += 1
                        pbar.update(1)
                        pbar.refresh()
                        
                        if len(dendrogram) in milestones and milestones[len(dendrogram)] is None:
                            elapsed = time.time() - start_time
                            milestones[len(dendrogram)] = elapsed
                            print(f"Reached {int((len(dendrogram)/total_merges)*100)}% in {elapsed:.2f} seconds")
                            
                        if U.n_subsets == 1: break
                        
            except StopIteration: pass

            try:
                dist, _ = P[i][1].peek()
                heapq.heappush(M, (dist, i))
            except StopIteration: pass

    assert len(dendrogram) == total_merges
    return dendrogram, milestones
def HNSW_HSSL_robust(data, ef=20, **hnsw_kwargs):
    
    M = []
    N = len(data)
    P = []
    
    start_time = time.time()
    total_merges = N - 1
    milestones = {
        int(total_merges * 0.25): None,
        int(total_merges * 0.5): None,
        int(total_merges * 0.75): None,
        int(total_merges * 0.8): None,
        int(total_merges * 0.9): None,
        int(total_merges * 0.95): None,
        int(total_merges * 0.99): None,
    }

    dendrogram = []

    graphs = gib.PyHNSW(data, **hnsw_kwargs)

    U = DisjointSet(np.arange(len(data)))
    cluster_id_map = np.arange(N)
    cluster_id_counter = N

    for i in tqdm(range(N), desc="Initializing"):
        searcher = PrioritySearcher_HNSW(data, graphs, U, i, ef=ef)
        P.append((searcher, peekable(searcher.advance_gen())))
        
        dist, _ = P[i][1].peek()
        heapq.heappush(M, (dist, i))


    with tqdm(total=N-1,desc="Merging clusters") as pbar:

        while U.n_subsets != 1:
            
            
            d, i = heapq.heappop(M)
            try:
                d2, j = P[i][1].__next__()
                if d == d2:
                    if not U.connected(i, j):

                        # add (d, i, j) to the dendrogram
                        dendrogram.append((
                            cluster_id_map[U[i]],
                            cluster_id_map[U[j]],
                            d,
                            U.subset_size(i) + U.subset_size(j),
                        ))
                        U.merge(i, j)
                        cluster_id_map[U[i]] = cluster_id_counter
                        cluster_id_counter += 1
                        pbar.update(1)
                        pbar.refresh()
                        
                        if len(dendrogram) in milestones and milestones[len(dendrogram)] is None:
                            elapsed = time.time() - start_time
                            milestones[len(dendrogram)] = elapsed
                            print(f"Reached {int((len(dendrogram)/total_merges)*100)}% in {elapsed:.2f} seconds")
                            
                        if U.n_subsets == 1: break
                        
            except StopIteration: pass

            try:
                dist, _ = P[i][1].peek()
                heapq.heappush(M, (dist, i))
            except StopIteration: pass

    assert len(dendrogram) == total_merges
    return dendrogram, milestones

# %% #### Clustering quality analysis
def find_cut(dendrogram):
	# https://github.com/sharpenb/Hierarchical-Paris-Clustering/
	from python_paris.homogeneous_cut_slicer import best_homogeneous_cut
	cut_level, cut_score = best_homogeneous_cut(np.array(dendrogram))
	# print(cut_level, cut_score)
	return len(dendrogram) + 1 - cut_level
def get_clustering_from_dendrogram(dendrogram, k, min_cluster_size=None):
    if min_cluster_size is None: min_cluster_size = 10
    # roundup_fix_dendrogram(dendrogram)
    N = len(dendrogram) + 1
    clustering = -np.ones(N+len(dendrogram), dtype=int)
    # Cluster ID -> merge distance order
    merge_distance_order = np.argsort([v[2] for v in dendrogram], kind="stable")
    # Merge distance order -> cluster ID
    inv_distance_order = merge_distance_order.copy()
    for i,j in enumerate(merge_distance_order): inv_distance_order[j] = i
    # Start with last merge, i.e. Cluster ID "N-1"
    final_clusters = set([len(dendrogram)-1])
    # i = N-2
    while len(final_clusters) < k and len(final_clusters) > 0:
        # Get Cluster ID of highest available merge distance
        i = inv_distance_order[max(merge_distance_order[v] for v in final_clusters)]
        final_clusters.remove(i)
        l,r = dendrogram[i][:2]
        l_size = 1 if l<N else dendrogram[l-N][3]
        r_size = 1 if r<N else dendrogram[r-N][3]
        if l >= N and l_size >= min_cluster_size: final_clusters.add(l-N)
        if r >= N and r_size >= min_cluster_size: final_clusters.add(r-N)
    if len(final_clusters) < k: print(f"Warning: Only found {len(final_clusters)} clusters, but should find {k}.")
    final_clusters = np.sort([*final_clusters]) + N
    for i,c in enumerate(final_clusters): clustering[c] = i
    for merged_index_diff, (cluster_i, cluster_j, _, _) in enumerate(reversed(dendrogram)):
        merged_index = N + len(dendrogram) - merged_index_diff - 1
        if merged_index > final_clusters[-1]: continue
        clustering[[cluster_i, cluster_j]] = clustering[merged_index]
    # print(np.unique(clustering[:N],return_counts=True))
    return clustering[:N]
# clustering = get_clustering_from_dendrogram(dendrogram, 5, 5)
# print(np.unique(clustering, return_counts=True))
# clustering
def ARI_score(dendro1, dendro2, min_cluster_size=None, k_override=None): 
        
        dendro1 = [[int(l), int(r), float(d), int(s)] for l, r, d, s in dendro1]
        dendro2 = [[int(l), int(r), float(d), int(s)] for l, r, d, s in dendro2]

        dendro1 = elki_sort_dendrogram(dendro1)
        dendro2 = elki_sort_dendrogram(dendro2)
        
        if k_override is None: k = find_cut(dendro1)
        else: k = k_override

        C1 = get_clustering_from_dendrogram(dendro1, k, min_cluster_size=min_cluster_size)
        C2 = get_clustering_from_dendrogram(dendro2, k, min_cluster_size=min_cluster_size)

        return adjusted_rand_score(C1, C2)

# %%
