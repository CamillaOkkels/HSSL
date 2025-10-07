use foldhash::HashSet;
use graphidx::{data::MatrixDataSource, graphs::{Graph, WeightedGraph}, heaps::MinHeap, measures::Distance, types::{SyncFloat, SyncUnsignedInteger}};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder};

use std::cmp::Reverse;
use priority_queue::PriorityQueue;
use ordered_float::OrderedFloat;
use std::fmt::Debug;
use num_traits::Float;
use std::time::{Instant, Duration};
use crate::hnsw::HNSWHeapBuildGraph;
use genawaiter::rc::{Gen, Co};
/* use crate::utils::union_find::DisjointSet; */
/* use std::simd::{f32x8, SimdFloat};

#[inline(always)]
fn l2_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = f32x8::splat(0.0);

    // process 8 elements per iteration
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let va = f32x8::from_slice(&a[i * 8..(i + 1) * 8]);
        let vb = f32x8::from_slice(&b[i * 8..(i + 1) * 8]);
        let diff = va - vb;
        sum += diff * diff;
    }

    // reduce SIMD sum
    let mut total = sum.reduce_sum();

    // handle remainder
    for i in (chunks * 8)..a.len() {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
} */

struct ObservedEdgesStore<R: SyncUnsignedInteger> {
	observed_edges: Vec<HashSet<R>>,
}
impl<R: SyncUnsignedInteger> ObservedEdgesStore<R> {
	fn new(n_elements: usize) -> Self {
		Self { observed_edges: vec![HashSet::default(); n_elements] }
	}
	fn observe_edge(&mut self, i: R, j: R) -> bool {
		if i != j {
			let (i,j) = (i.min(j), i.max(j));
			unsafe{self.observed_edges.get_unchecked_mut(i.to_usize().unwrap_unchecked()).insert(j)}
		} else { false }
	}
}
struct UnionFind<R: SyncUnsignedInteger> {
	parents: Vec<R>,
    sizes: Vec<usize>,  // track subset sizes
    n_subsets: usize,   // track number of subsets
}
impl<R: SyncUnsignedInteger> UnionFind<R> {
	fn new(n_elements: usize) -> Self {
		assert!(R::max_value().to_usize().unwrap() >= n_elements);
		assert!(isize::max_value() as usize >= n_elements);
		let mut parents = Vec::with_capacity(n_elements);
		unsafe{parents.extend((0..n_elements).map(|i| R::from(i).unwrap_unchecked()));}
		Self { 
		parents,
		sizes: vec![1; n_elements],
		n_subsets: n_elements
		}
	}
	fn find_immutable(&self, i: R) -> R {
		unsafe {
			let parents = self.parents.as_ptr();
			let mut i = i;
			let mut par = parents.offset(i.to_isize().unwrap());
			while *par != i {
				(i, par) = (*par, parents.offset(i.to_isize().unwrap()));
			}
			i
		}
	}
	/* Gets the root node of the union find and returns it */
	fn find(&mut self, i: R) -> R {
		unsafe {
			let parents = self.parents.as_mut_ptr();
			let mut i = i;
			let mut par = parents.offset(i.to_isize().unwrap());
			while *par != i {
				/* Path splitting */
				let next_par = parents.offset((*par).to_isize().unwrap());
				(i, *par, par) = (*par, *next_par, next_par);
			}
			i
		}
	}
	/* Combines two sets by making the root of the first set the child
	 * of the second set */
    fn union(&mut self, i: R, j: R) {
        let i_root = self.find(i);
        let j_root = self.find(j);

        if i_root != j_root {
            let i_usize = i_root.to_usize().unwrap();
            let j_usize = j_root.to_usize().unwrap();

            // Merge smaller tree into larger tree
            if self.sizes[i_usize] < self.sizes[j_usize] {
                // i_root becomes child of j_root
                self.parents[i_usize] = j_root;
                self.sizes[j_usize] += self.sizes[i_usize];
                self.sizes[i_usize] = 0;
            } else {
                // j_root becomes child of i_root
                self.parents[j_usize] = i_root;
                self.sizes[i_usize] += self.sizes[j_usize];
                self.sizes[j_usize] = 0;
            }

            self.n_subsets -= 1;
        }
    }
	/* fn union(&mut self, i: R, j: R) {
		let i_root = self.find(i);
		let j_root = self.find(j);
		self.parents[i_root.to_usize().unwrap()] = j_root;
	} */
	fn n_subsets(&self) -> usize {
    	self.n_subsets
    }

    fn subset_size(&mut self, i: R) -> usize {
        let root = self.find(i);
        self.sizes[root.to_usize().unwrap()]
    }
	
	// Returns true if i and j are in the same subset
    pub fn connected(&mut self, i: R, j: R) -> bool {
        self.find(i) == self.find(j)
    }
}


pub fn is_hnsw_connected<R, F, Dist>(hnsw: &HNSWParallelHeapBuilder<R, F, Dist>) -> bool
where
    R: SyncUnsignedInteger + Copy,
    F: SyncFloat,
    Dist: Distance<F> + Sync + Send,
{
    let n = hnsw.n_data;
    if n == 0 {
        return true;
    }

    // Track visited global nodes
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    // Start from node 0
    visited[0] = true;
    queue.push_back(R::from_usize(0).unwrap());

    while let Some(current_global) = queue.pop_front() {
        let _current_usize = current_global.to_usize().unwrap();

        for (layer_idx, graph) in hnsw.graphs.iter().enumerate() {
            // Resolve local ID from global
            let local_id_opt = if layer_idx == 0 {
                Some(current_global) // no mapping needed
            } else {
                // Lookup local ID in reverse (global_id -> local_id)
                let id_map = &hnsw.global_layer_ids[layer_idx - 1];
                id_map.iter().position(|&gid| gid == current_global).map(|i| R::from_usize(i).unwrap())
            };

            if let Some(local_id) = local_id_opt {
                graph.foreach_neighbor(local_id, |&neighbor_local| {
                    let neighbor_global = if layer_idx == 0 {
                        neighbor_local
                    } else {
                        let id_map = &hnsw.global_layer_ids[layer_idx - 1];
                        *unsafe { id_map.get_unchecked(neighbor_local.to_usize().unwrap()) }
                    };

                    let neighbor_usize = neighbor_global.to_usize().unwrap();
                    if !visited[neighbor_usize] {
                        visited[neighbor_usize] = true;
                        queue.push_back(neighbor_global);
                    }
                });
            }
        }
    }

    visited.iter().all(|&v| v)
}
	// let connected = is_hnsw_connected(&hnsw);
	// println!("Graph is connected: {}", connected);
	/*if !is_hnsw_connected(&hnsw, n) {
    println!("Warning: HNSW graph is not fully connected!");
    // Optionally: return or visualize disconnected components
	}*/
	
pub fn graph_based_dendrogram<
	F: SyncFloat,
	R: SyncUnsignedInteger,
	M: MatrixDataSource<F>+graphidx::types::Sync,
	Dist: Distance<F>+Sync+Send,
// >(data: &M, dist: Dist, min_pts: usize, expand: bool, symmetric_expand: bool, hnsw_params: HNSWParams<F>) -> (Vec<(usize, usize, F, usize)>, Vec<F>) {
>(data: &M, dist: Dist, min_pts: usize, expand: bool, symmetric_expand: bool, hnsw_params: HNSWParams<F>) -> (Vec<(usize, usize, F, usize)>, Vec<F>, Vec<std::time::Duration>) {
	let n = data.n_rows();
	assert!(R::max_value().to_usize().unwrap() >= n);

	let start_time = Instant::now();
	let mut next_milestone_idx = 0;
	let milestone_steps = [0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99].map(|p| ((n - 1) as f64 * p).round() as usize);
	let mut milestones = Vec::with_capacity(milestone_steps.len());

	/* Build HNSW on the data and get the graphs */
	let hnsw = HNSWParallelHeapBuilder::<R,_,_>::base_init(data, dist, hnsw_params);
	let (graphs, _, global_ids, dist) = hnsw._into_parts();
	/* Create storage for observed edges and accessor function */
	let mut observed_edges = ObservedEdgesStore::new(n);
	/* Create an expand queue for edges and insert all edges from all graph levels */
	let mut expand_queue = MinHeap::with_capacity(graphs.iter().map(|g| g.n_edges()).sum::<usize>() + 1_000);
	(0..graphs.len()).for_each(|i| {
		let graph = &graphs[i];
		if i > 0 { /* Higher layers */
			let id_map = global_ids.get(i-1).unwrap();
			(0..graph.n_vertices()).for_each(|i_node| {
				let i_global = *unsafe{id_map.get_unchecked(i_node)};
				let i_node = unsafe{R::from(i_node).unwrap_unchecked()};
				graph.foreach_neighbor_with_zipped_weight(i_node, |&d, &j_node| {
					let j_global = *unsafe{id_map.get_unchecked(j_node.to_usize().unwrap_unchecked())};
					if observed_edges.observe_edge(i_global, j_global) {
						expand_queue.push(d, (i_global.min(j_global), i_global.max(j_global)));
					}
				});
			});
		} else { /* Bottom layer */
			(0..graph.n_vertices()).for_each(|i_node| {
				let i_node = unsafe{R::from(i_node).unwrap_unchecked()};
				graph.foreach_neighbor_with_zipped_weight(i_node, |&d, &j_node| {
					if observed_edges.observe_edge(i_node, j_node) {
						expand_queue.push(d, (i_node.min(j_node), i_node.max(j_node)));
					}
				});
			});
		}
	});
	/* Get bottom layer graph for all further operations
	 * and translate into a more efficient condensed format */
	let graph = graphs.get(0).unwrap().as_dir_lol_graph();
	// let graph = graphs.get(0).unwrap().to_owned();
	// let start_time = std::time::Instant::now();
	/* Create a union find structure for the clusters */
	let mut union_find: UnionFind<R> = UnionFind::new(n);
	/* Create a storage for the dendrogram cluster ids */
	let mut cluster_ids = (0..n).collect::<Vec<usize>>();
	/* Create a storage for the cluster sizes */
	let mut cluster_sizes = vec![1; n];
	/* Create a storage for the dendrogram */
	let mut dendrogram = Vec::with_capacity(n - 1);
	/* Create a storage to count the number of visited edges for each node
	 * and a storage for the distance at which any point becomes a core point */
	let neighbor_counts = vec![0usize; n];
	let mut neighbor_stacks = (0..n).map(|_| MinHeap::<F,R>::with_capacity(min_pts-1)).collect::<Vec<_>>();
	let mut core_distances = vec![-F::one(); n];
	let mut core_point_count = 0;
	/* Do the actual clustering */
	unsafe {
		/* Shorthand to merge two clusters */
		let mut merge_clusters = |
			dendrogram: &mut Vec<(usize,usize,F,usize)>,
			union_find: &mut UnionFind<R>,
			cluster_ids: &mut Vec<usize>,
			cluster_sizes: &mut Vec<usize>,
			distance: F, i_root: usize, j_root: usize
		| {
			/* Update dendrogram info */
			let i_cluster_id = cluster_ids.get_unchecked(i_root);
			let j_cluster_id = cluster_ids.get_unchecked(j_root);
			let i_cluster_size = cluster_sizes.get_unchecked(i_root);
			let j_cluster_size = cluster_sizes.get_unchecked(j_root);
			let new_id = n + dendrogram.len();
			let new_size = *i_cluster_size + *j_cluster_size;
			dendrogram.push((*i_cluster_id, *j_cluster_id, distance, new_size));
			/* Update union find and cluster infos */
			union_find.union(R::from(i_root).unwrap_unchecked(), R::from(j_root).unwrap_unchecked());
			let new_root = j_root;
			*cluster_ids.get_unchecked_mut(new_root) = new_id;
			*cluster_sizes.get_unchecked_mut(new_root) = new_size;

			if next_milestone_idx < milestone_steps.len() && dendrogram.len() >= milestone_steps[next_milestone_idx] {
				milestones.push(start_time.elapsed());
				next_milestone_idx += 1;
			}

		};
		while dendrogram.len() < n-1 && expand_queue.size() > 0 {
			/* Get the next edge */
			let (d_ij, (i, j)) = expand_queue.pop().unwrap_unchecked();
			let i_usize = i.to_usize().unwrap_unchecked();
			let j_usize = j.to_usize().unwrap_unchecked();
			/* Update core point info */
			let both_core_points = vec![(i, i_usize, j), (j, j_usize, i)].iter().map(|&(idx, idx_usize, other_idx)| {
				let cnt = neighbor_counts.as_ptr().offset(idx_usize as isize) as *mut usize;
				*cnt += 1;
				if *cnt==min_pts {
					*core_distances.get_unchecked_mut(idx_usize) = d_ij;
					core_point_count += 1;
					/* Get root objects */
					let root = union_find.find(idx).to_usize().unwrap_unchecked();
					/* Attempt to merge with all neighbors first */
					let neighbor_stack = neighbor_stacks.get_unchecked_mut(idx_usize);
					while neighbor_stack.size() > 0 {
						let (distance,other) = neighbor_stack.pop().unwrap_unchecked();
						if *neighbor_counts.get_unchecked(other.to_usize().unwrap_unchecked()) >= min_pts {
							let other_root = union_find.find(other).to_usize().unwrap_unchecked();
							if root != other_root {
								merge_clusters(
									&mut dendrogram,
									&mut union_find,
									&mut cluster_ids,
									&mut cluster_sizes,
									distance, other_root, root
								);
							}
						}
					}
				} else if *cnt < min_pts {
					neighbor_stacks.get_unchecked_mut(idx_usize).push(d_ij,other_idx);
				}
				*cnt >= min_pts
			}).all(|b| b);
			/* Merge clusters if both are core points now */
			if both_core_points {
				/* Get root objects again, in case they changed due to ops before */
				let i_root = union_find.find(i).to_usize().unwrap_unchecked();
				let j_root = union_find.find(j).to_usize().unwrap_unchecked();
				if i_root == j_root { continue; }
				/* Update dendrogram info */
				merge_clusters(
					&mut dendrogram,
					&mut union_find,
					&mut cluster_ids,
					&mut cluster_sizes,
					d_ij, i_root, j_root
				);
			}
			/* Expand on the edge and add new entries to the expand queue
			 * unless that is disabled with `expand = false` */
			if expand {
				/* Compute pairwise distances between neighborhoods in parallel.
				* Skip loops and already merged pairs. */
				let work_output = if symmetric_expand {
					/* Extend the expand queue with neighbor-of-neighbor pairs */
					let nodes1: Vec<R> = graph.iter_neighbors(i).cloned().chain(std::iter::once(i)).collect();
					let nodes2: Vec<R> = graph.iter_neighbors(j).cloned().chain(std::iter::once(j)).collect();
					let total_work = nodes1.len() * nodes2.len();
					let n_threads = rayon::current_num_threads();
					let work_per_thread = (total_work+n_threads-1) / n_threads;
					let mut work_output: Vec<(F,(R,R))> = Vec::with_capacity(total_work);
					work_output.set_len(total_work);
					work_output.chunks_mut(work_per_thread).enumerate().collect::<Vec<_>>().into_par_iter().for_each(|(i_thread, output)| {
						let start = i_thread * work_per_thread;
						let end = std::cmp::min(start + work_per_thread, total_work);
						(start..end).enumerate().for_each(|(i_output, i_job)| {
							let output_cell = output.get_unchecked_mut(i_output);
							let i = nodes1[i_job / nodes2.len()];
							let j = nodes2[i_job % nodes2.len()];
							if i == j {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_root = union_find.find_immutable(i);
							let j_root = union_find.find_immutable(j);
							if i_root == j_root {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_usize = i.to_usize().unwrap_unchecked();
							let j_usize = j.to_usize().unwrap_unchecked();
							let i_row = data.get_row_view(i_usize);
							let j_row = data.get_row_view(j_usize);
							let d_ij = dist.dist_slice(&i_row, &j_row);
							*output_cell = (d_ij, (i.min(j), i.max(j)));
						});
					});
					work_output
				} else {
					/* Extend the expand queue with neighbor-of-neighbor pairs */
					let nodes1: Vec<R> = graph.iter_neighbors(i).cloned().collect();
					let nodes2: Vec<R> = graph.iter_neighbors(j).cloned().collect();
					let total_work = nodes1.len() + nodes2.len();
					let n_threads = rayon::current_num_threads();
					let work_per_thread = (total_work+n_threads-1) / n_threads;
					let mut work_output: Vec<(F,(R,R))> = Vec::with_capacity(total_work);
					work_output.set_len(total_work);
					work_output.chunks_mut(work_per_thread).enumerate().collect::<Vec<_>>().into_par_iter().for_each(|(i_thread, output)| {
						let start = i_thread * work_per_thread;
						let end = std::cmp::min(start + work_per_thread, total_work);
						(start..end).enumerate().for_each(|(i_output, i_job)| {
							let output_cell = output.get_unchecked_mut(i_output);
							let (i,j) = if i_job < nodes1.len() {
								(nodes1[i_job], j)
							} else {
								(i, nodes2[i_job - nodes1.len()])
							};
							if i == j {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_root = union_find.find_immutable(i);
							let j_root = union_find.find_immutable(j);
							if i_root == j_root {
								*output_cell = (-F::one(), (R::zero(), R::zero()));
								return;
							}
							let i_usize = i.to_usize().unwrap_unchecked();
							let j_usize = j.to_usize().unwrap_unchecked();
							let i_row = data.get_row_view(i_usize);
							let j_row = data.get_row_view(j_usize);
							let d_ij = dist.dist_slice(&i_row, &j_row);
							*output_cell = (d_ij, (i.min(j), i.max(j)));
						});
					});
					work_output
				};
				/* Insert edges into the queue if not yet observed */
				work_output.into_iter()
				.filter(|(d,_)| *d >= F::zero())
				.for_each(|(d_ij, (i, j))| {
					if observed_edges.observe_edge(i, j) {
						/* Only push if the edge was not already observed */
						expand_queue.push(d_ij, (i.min(j), i.max(j)));
					}
				});
			}
		}
	}
	// println!("{:?}", start_time.elapsed());
	/* Return the dendrogram */
	(dendrogram, core_distances, milestones)
}

/*
#[cfg(test)]
mod tests {
	use ndarray::Array2;
	use ndarray_rand::rand_distr::Normal;
	use rand::prelude::Distribution;

	use crate::{cluster::graph_based_dendrogram, hnsw::HNSWParams};

	#[test]
	fn test_graph_based_dendrogram() {
		let (n,d) = (10_000, 3);
		let rng1 = Normal::new(0.0, 1.0).unwrap();
		let rng2 = Normal::new(5.0, 1.0).unwrap();
		let data: Array2<f32> = Array2::from_shape_fn((n, d), |(i,_)| (if i%2 == 0 {rng1} else {rng2}).sample(&mut rand::thread_rng()));
		let start_time = std::time::Instant::now();
		let (dendrogram, core_distances) = graph_based_dendrogram::<f32, u32, _, _>(
			&data,
			graphidx::measures::SquaredEuclideanDistance::new(),
			5,
			true,
			HNSWParams::new(),
		);
		println!("Dendrogram computation took: {:.2?}", start_time.elapsed());
		dendrogram.iter().skip(n-10).for_each(|v| {
			println!("{} {} {} {}", v.0, v.1, v.2, v.3);
		});
		let mean_core_dist = core_distances.iter().sum::<f32>() / core_distances.len() as f32;
		let std_core_dist = core_distances.iter().map(|&x| (x - mean_core_dist).powi(2)).sum::<f32>() / core_distances.len() as f32;
		println!("Mean core distance: {}", mean_core_dist);
		println!("Std core distance: {}", std_core_dist);
	}
} */


// -------------- HNSW HSSL implementation -------------- //

// ---- HNSW-HSSL Priority Searcher ---- ////
pub struct PrioritySearcherHnsw<'a, M, F, R, Dist>
where
    M: MatrixDataSource<F> + Sync,
    F: SyncFloat + Debug + Copy + PartialOrd,
    R: SyncUnsignedInteger + Copy,
    Dist: Distance<F> + Sync + Send,
{
    data: &'a M,
    graph: &'a HNSWHeapBuildGraph<R, F>,
    dist: &'a Dist, // store reference to distance impl
    query: usize,
    expand_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
    candidate_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
    visited: HashSet<usize>,
    ef: usize,
}

impl<'a, M, F, R, Dist> PrioritySearcherHnsw<'a, M, F, R, Dist>
where
    M: MatrixDataSource<F> + Sync,
    F: SyncFloat + Debug + Copy + PartialOrd,
    R: SyncUnsignedInteger + Copy,
    Dist: Distance<F> + Sync + Send,
{
    pub fn new(
        data: &'a M,
        graph: &'a HNSWHeapBuildGraph<R, F>,
        dist: &'a Dist,
        query: usize,
        ef: usize,
        union_find: &'a mut UnionFind<usize>, // new change
    ) -> Self {
        Self {
            data,
            graph,
            dist,
            query,
            ef,
            expand_queue: PriorityQueue::new(),
            candidate_queue: PriorityQueue::new(),
            visited: HashSet::default(),
        }
    }

    // Use the Distance trait instead of ndarray arithmetic.
    /* fn batch_dists(&self, js: &[usize]) -> Vec<F> {
        let qv = self.data.get_row_view(self.query);
        js.iter() // into_par_iter()
            .map(|&j| {
                let jv = self.data.get_row_view(j);
                // dist_slice expects slices/views as provided by MatrixDataSource
                self.dist.dist_slice(&qv, &jv)
            })
            .collect()
    } // zip with mutable vec */

	fn batch_dists(&self, js: &[usize], out: &mut [F]) {
    let qv = self.data.get_row_view(self.query);
    for (j, slot) in js.iter().zip(out.iter_mut()) {
        let jv = self.data.get_row_view(*j);
        *slot = self.dist.dist_slice(&qv, &jv); // l2_simd(qv, jv); 
    	}
	}

    /* pub fn advance_gen(&mut self, union_find: &mut UnionFind<usize>, return_self: bool) -> Vec<(F, usize)>{
        let mut results = Vec::new();

        if return_self {
            results.push((F::zero(), self.query));
        }

        self.expand_queue.clear();
        self.expand_queue.push(self.query, Reverse(OrderedFloat(F::zero())));
        self.candidate_queue.clear();
        self.visited.clear();
        self.visited.insert(self.query);

        while let Some((expansion_point, Reverse(ord_dist))) = self.expand_queue.pop() {
            // ord_dist is OrderedFloat<F>
            let mut neighbors: Vec<usize> = self.graph.neighbors(R::from_usize(expansion_point).unwrap()).into_iter().map(|r| r.to_usize().unwrap()).collect();
            neighbors.retain(|n| !self.visited.contains(n));

            // filter out neighbors already in same union-find subset
            let mut reduced_neighbors = Vec::new();
            for &n in &neighbors {
                if !union_find.connected(self.query, n) {
                    reduced_neighbors.push(n);
                } else {
                    self.visited.insert(n);
                }
            }
            neighbors = reduced_neighbors;

			let mut neighbor_distances = vec![F::zero(); neighbors.len()];
			self.batch_dists(&neighbors, &mut neighbor_distances);
            for (&neighbor, &distance) in neighbors.iter().zip(&neighbor_distances) {
                self.visited.insert(neighbor);
                self.expand_queue.push(neighbor, Reverse(OrderedFloat(distance)));
                self.candidate_queue.push(neighbor, Reverse(OrderedFloat(distance)));
            }

            // reduce candidate queue to ef by popping largest distance candidates
            while self.candidate_queue.len() > self.ef {
                if let Some((idx, Reverse(ord_dist))) = self.candidate_queue.pop() {
                    results.push((ord_dist.into_inner(), idx));
                } else {
                    break;
                }
            }
        }

        while let Some((idx, Reverse(ord_dist))) = self.candidate_queue.pop() {
            results.push((ord_dist.into_inner(), idx));
        }

        results
    } */
	
    pub fn advance_gen<'b>(
        &'b mut self,
        union_find: &mut UnionFind<usize>,
        return_self: bool,
    ) -> impl Iterator<Item = (F, usize)> + 'b {
        Gen::<(F, usize), (), _>::new(|co: Co<(F, usize), ()>| async move {
            if return_self {
                co.yield_((F::zero(), self.query)).await;
            }

            self.expand_queue.clear();
            self.expand_queue.push(self.query, Reverse(OrderedFloat(F::zero())));
            self.candidate_queue.clear();
            self.visited.clear();
            self.visited.insert(self.query);

            while let Some((expansion_point, Reverse(ord_dist))) = self.expand_queue.pop() {
                let mut neighbors: Vec<usize> = self.graph
                    .neighbors(R::from_usize(expansion_point).unwrap())
                    .into_iter()
                    .map(|r| r.to_usize().unwrap())
                    .collect();
                neighbors.retain(|n| !self.visited.contains(n));

                let mut reduced_neighbors = Vec::new();
                for &n in &neighbors {
                    if !self.union_find.connected(self.query, n) {
                        reduced_neighbors.push(n);
                    } else {
                        self.visited.insert(n);
                    }
                }
                neighbors = reduced_neighbors;

                let mut neighbor_distances = vec![F::zero(); neighbors.len()];
                self.batch_dists(&neighbors, &mut neighbor_distances);

                for (&neighbor, &distance) in neighbors.iter().zip(&neighbor_distances) {
                    self.visited.insert(neighbor);
                    self.expand_queue.push(neighbor, Reverse(OrderedFloat(distance)));
                    self.candidate_queue.push(neighbor, Reverse(OrderedFloat(distance)));
                }

                while self.candidate_queue.len() > self.ef {
                    if let Some((idx, Reverse(ord_dist))) = self.candidate_queue.pop() {
                        co.yield_((ord_dist.into_inner(), idx)).await;
                    } else {
                        break;
                    }
                }
            }

            while let Some((idx, Reverse(ord_dist))) = self.candidate_queue.pop() {
                co.yield_((ord_dist.into_inner(), idx)).await;
            }
        }).into_iter()
    }

}

// ---- HNSW-HSSL clustering function ---- //
/* pub fn hnsw_hssl<
	F: SyncFloat,
	R: SyncUnsignedInteger,
	M: MatrixDataSource<F> + graphidx::types::Sync,
	Dist: Distance<F>+Sync+Send,
>(data: &M, dist: Dist, ef: &R, hnsw_params: HNSWParams<F>) -> (Vec<(usize, usize, F, usize)>, Vec<std::time::Duration>) 
{
    let n = data.n_rows();
    let start_time = std::time::Instant::now();
	let mut next_milestone_idx = 0;
	let milestone_steps = [0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99].map(|p| ((n - 1) as f64 * p).round() as usize);
	let mut milestones = Vec::with_capacity(milestone_steps.len());

    let mut dendrogram = Vec::with_capacity(n - 1);

    let mut union_find: UnionFind<usize> = UnionFind::new(n);
    let mut cluster_id_map: Vec<usize> = (0..n).collect();
    let mut cluster_id_counter = n;

	/* Build HNSW on the data and get the graphs */
	let hnsw = HNSWParallelHeapBuilder::<R,_,_>::base_init(data, dist, hnsw_params);
	let (graphs, _, global_ids, dist) = hnsw._into_parts();
	let bottom_layer = &graphs[0];

	// inside hnsw_hssl(...) after you have `dist` and `bottom_layer`:
	let ef_usize = ef.to_usize().unwrap(); // convert ef (R) -> usize

	let mut candidate_heap: PriorityQueue<usize, Reverse<OrderedFloat<F>>> = PriorityQueue::new();
	let mut searchers: Vec<PrioritySearcherHnsw<_, F, _, _>> = Vec::with_capacity(n);

	/* println!("test 1"); */

	for i in 0..n {
		println!("making searcher = {}", i);
		let mut searcher = PrioritySearcherHnsw::new(data, bottom_layer, &dist, i, ef_usize);
		let mut advance_results = searcher.advance_gen(&mut union_find, false);
		println!("searcher {} created", i);
		if let Some(&(dist0, _)) = advance_results.next() {
			candidate_heap.push(i, Reverse(OrderedFloat(dist0)));
		}
		searchers.push(searcher);
	}
	/* println!("test 3"); */


    // Main merging loop
    while union_find.n_subsets() > 1 {
        // Pop from heap min distance (priority queue stores OrderedFloat, smaller = higher priority)
        let (i, Reverse(ord_d)) = candidate_heap.pop().expect("Priority queue should not be empty");
        let d = ord_d.into_inner();

		println!("{} clusters", union_find.n_subsets());

        // get the advance results for this searcher given current union-find state
        let advance_results = {
            let searcher = &mut searchers[i];
            searcher.advance_gen(&mut union_find, false)
        };

        for &(dist2, j) in advance_results {
            if dist2 == d && !union_find.connected(i, j) {
                dendrogram.push((
                    cluster_id_map[union_find.find(i)],
                    cluster_id_map[union_find.find(j)],
                    d,
                    union_find.subset_size(i) + union_find.subset_size(j),
                ));
                union_find.union(i, j);
                cluster_id_map[union_find.find(i)] = cluster_id_counter;
                cluster_id_counter += 1;

				if next_milestone_idx < milestone_steps.len() && dendrogram.len() >= milestone_steps[next_milestone_idx] {
					milestones.push(start_time.elapsed());
					next_milestone_idx += 1;
				}

                if union_find.n_subsets() == 1 {
                    break;
                }
            }
        }

        // Push next round: find next candidate distance > d
        if let Some(&(dist_next, _)) = advance_results.clone().find(|&&(dist2, _)| dist2 > d) {
            candidate_heap.push(i, Reverse(OrderedFloat(dist_next)));
        }
    }

    (dendrogram, milestones)
} */
pub fn hnsw_hssl<F, R, M, Dist>(
    data: &M,
    dist: Dist,
    ef: &R,
    hnsw_params: HNSWParams<F>,
) -> (Vec<(usize, usize, F, usize)>, Vec<std::time::Duration>)
where
    F: SyncFloat + Debug + Copy + PartialOrd,
    R: SyncUnsignedInteger + Copy,
    M: MatrixDataSource<F> + graphidx::types::Sync,
    Dist: Distance<F> + Sync + Send,
{
    let n = data.n_rows();
    let start_time = Instant::now();

    let mut dendrogram = Vec::with_capacity(n - 1);

    // Milestone tracking
    let total_merges = n - 1;
    let milestone_steps = [0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
        .map(|p| ((total_merges as f64) * p).round() as usize);
    let mut milestones = Vec::with_capacity(milestone_steps.len());
    let mut next_milestone_idx = 0;

    // Build HNSW
    let hnsw = HNSWParallelHeapBuilder::<R, _, _>::base_init(data, dist, hnsw_params);
    let (graphs, _, global_ids, dist) = hnsw._into_parts();
    let bottom_layer = &graphs[0];

    let ef_usize = ef.to_usize().unwrap();

    // Initialize searchers and candidate heap
    let mut searchers: Vec<_> = Vec::with_capacity(n);
    let mut candidate_heap: PriorityQueue<usize, Reverse<OrderedFloat<F>>> = PriorityQueue::new();
    let mut advance_iters: Vec<_> = Vec::with_capacity(n);

    let mut union_find = UnionFind::<usize>::new(n);
    let mut cluster_id_map: Vec<usize> = (0..n).collect();
    let mut cluster_id_counter = n;

    for i in 0..n {
        let mut searcher = PrioritySearcherHnsw::new(data, bottom_layer, &dist, i, ef_usize, &mut union_find);
        let mut advance_iter = searcher.advance_gen(&mut UnionFind<usize>, false).peekable();

        // Push first distance into candidate heap (Python peek)
        if let Some(&(dist0, _)) = advance_iter.peek() {
            candidate_heap.push(i, Reverse(OrderedFloat(dist0)));
        }

        searchers.push(searcher);
        advance_iters.push(advance_iter);
    }

    // Main HSSL loop
    while union_find.n_subsets() > 1 {
        let (i, Reverse(ord_d)) = candidate_heap.pop().expect("Candidate heap empty");
        let d = ord_d.into_inner();

        let advance_iter: &mut Gen<(F, usize), (), _> = &mut advance_iters[i];
        // Get next advance result for this searcher
        while let Some((dist2, j)) = advance_iter.next() {
            if dist2 == d && !union_find.connected(i, j) {
                // Merge clusters
                let i_root = union_find.find(i);
                let j_root = union_find.find(j);
                dendrogram.push((
                    cluster_id_map[i_root],
                    cluster_id_map[j_root],
                    d,
                    union_find.subset_size(i) + union_find.subset_size(j),
                ));
                union_find.union(i, j);
                let new_root = union_find.find(i);
                cluster_id_map[new_root] = cluster_id_counter;
                cluster_id_counter += 1;

                // Milestone tracking
                if next_milestone_idx < milestone_steps.len()
                    && dendrogram.len() >= milestone_steps[next_milestone_idx]
                {
                    milestones.push(start_time.elapsed());
                    next_milestone_idx += 1;
                }

                if union_find.n_subsets() == 1 {
                    break;
                }
            }
            // Only push next candidate distance greater than current `d`
            if dist2 > d {
                candidate_heap.push(i, Reverse(OrderedFloat(dist2)));
                break;
            }
        }
    }

    (dendrogram, milestones)
}

#[test]
fn test_hnsw_hssl() {
    use graphidx::measures::SquaredEuclideanDistance;
	use ndarray::Array2;
	use ndarray_rand::rand_distr::Normal;
	use rand::prelude::Distribution;

    let (n, d) = (10_000, 3);
    let rng1 = Normal::new(0.0, 1.0).unwrap();
    let rng2 = Normal::new(5.0, 1.0).unwrap();
    let data: Array2<f32> = Array2::from_shape_fn((n, d), |(i, _)| {
        (if i % 2 == 0 { rng1 } else { rng2 })
            .sample(&mut rand::thread_rng())
    });

    let start_time = std::time::Instant::now();
    let (dendrogram, milestones) = crate::cluster::hnsw_hssl::<f32, u32, _, _>(
        &data,
        SquaredEuclideanDistance::new(),
        &100u32,  // ef parameter
        HNSWParams::new(),
    );
    println!("HNSW-HSSL computation took: {:.2?}", start_time.elapsed());

    // Check dendrogram length
    assert_eq!(dendrogram.len(), n - 1);

    // Print last 10 merges
    dendrogram.iter().skip(n - 10).for_each(|v| {
        println!("{} {} {} {}", v.0, v.1, v.2, v.3);
    });

    // Print milestones
    for (i, dur) in milestones.iter().enumerate() {
        println!("Milestone {}: {:.2?}", i, dur);
    }
}