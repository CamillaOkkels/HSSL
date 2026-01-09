use foldhash::HashSet;
use graphidx::{data::MatrixDataSource, graphs::{Graph, WeightedGraph}, heaps::MinHeap, measures::Distance, types::{SyncFloat, SyncUnsignedInteger}};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::{VecDeque, HashMap};

use crate::hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder};

use rand::Rng;
use rand::seq::SliceRandom;
use indicatif::{ProgressBar, ProgressStyle};
use std::cmp::Reverse;
use priority_queue::PriorityQueue;
use ordered_float::OrderedFloat;
use std::fmt::Debug;
// use num_traits::Float;
use std::time::{Instant, Duration};
use crate::hnsw::HNSWHeapBuildGraph;
use genawaiter::rc::{Gen};
use genawaiter::GeneratorState;
use std::future::Future;
use std::pin::Pin;
use std::cell::RefCell;
use std::rc::Rc;
// use std::sync::Arc;

#[derive(Clone)]
pub struct SquaredEuclideanDistance { /* fields if any */ }
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

	pub fn get_sets(&mut self) -> Vec<Vec<usize>> {
		let mut sets = Vec::with_capacity(self.n_subsets);
		let mut root_map = HashMap::<R, usize>::new();
		self.parents.iter().enumerate().for_each(|(i, &par)| {
			if i == par.to_usize().unwrap() {
				root_map.insert(par, root_map.len());
				sets.push(Vec::with_capacity(self.sizes[i]));
			}
		});
		for i in 0..self.parents.len() {
			let root = self.find(R::from(i).unwrap());
			sets[*root_map.get(&root).unwrap()].push(i);
		}
		sets
	}
}


fn is_connected_hnsw_full<R: SyncUnsignedInteger, F: SyncFloat>(
	hnsw: &HNSWParallelHeapBuilder<R, F, impl Distance<F> + Sync + Send>
) -> (bool, Vec<Vec<usize>>) {

	let n = hnsw.n_data;
	if n == 0 {
		(true, Vec::<Vec<usize>>::new());
	}

	// Initialize UnionFind
	let mut uf = UnionFind::<R>::new(n);

	// Build combined adjacency: for each global node, collect all its neighbors (global IDs)
	let mut adjacency: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::default()).collect();

	// Loop over layers
	for (layer_idx, graph) in hnsw._graphs().iter().enumerate() {
		if layer_idx == 0 {
			// Bottom layer: IDs are global
			for i in 0..n {
				let node = R::from_usize(i).unwrap();
				for &(_, nbr) in graph.view_neighbors_heap(node).iter() {
					let j = nbr.to_usize().unwrap();
					adjacency[i].insert(j);
					adjacency[j].insert(i);
					uf.union(node, R::from_usize(j).unwrap());
				}
			}
		} else {
			// Upper layers: need to map local → global
			let global_ids = &hnsw._global_layer_ids()[layer_idx - 1];
			for (local_idx, &global_i) in global_ids.iter().enumerate() {
				let node = R::from_usize(local_idx).unwrap();
				for &(_, nbr_local) in graph.view_neighbors_heap(node).iter() {
					let global_j = global_ids[nbr_local.to_usize().unwrap()];
					let i_usize = global_i.to_usize().unwrap();
					let j_usize = global_j.to_usize().unwrap();
					adjacency[i_usize].insert(j_usize);
					adjacency[j_usize].insert(i_usize);
					uf.union(global_i, global_j);
				}
			}
		}
	}

	// Collect connected components
	let mut components_map: HashMap<R, Vec<usize>> = HashMap::new();
	for i in 0..n {
		let root = uf.find(R::from_usize(i).unwrap());
		components_map.entry(root).or_default().push(i);
	}

	// Now BFS over combined adjacency
	let mut visited = vec![false; n];
	let mut queue = VecDeque::new();
	visited[0] = true;
	queue.push_back(0);

	while let Some(i) = queue.pop_front() {
		for &j in &adjacency[i] {
			if !visited[j] {
				visited[j] = true;
				queue.push_back(j);
			}
		}
	}

	let components: Vec<Vec<usize>> = components_map.into_values().collect();
	let is_connected = components.len() == 1;

	(is_connected, components)
}


fn random_sample_points(
    comp1: &Vec<usize>,
    comp2: &Vec<usize>,
    n: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut rng = rand::thread_rng();

    // Sample points from first component
    let sample1 = if comp1.len() <= n {
        comp1.clone()
    } else {
        comp1.choose_multiple(&mut rng, n).copied().collect()
    };

    // Sample points from second component
    let sample2 = if comp2.len() <= n {
        comp2.clone()
    } else {
        comp2.choose_multiple(&mut rng, n).copied().collect()
    };

    (sample1, sample2)
}

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

    // Build HNSW on the data and get the graphs 
    let hnsw = HNSWParallelHeapBuilder::<R,_,_>::base_init(data, dist, hnsw_params);

    let (connected, components) = is_connected_hnsw_full(&hnsw);
    /* println!("Graph is connected: {}", connected);
    println!("# of components: {}", components.len()); */

	let (graphs, _, global_ids, dist) = hnsw._into_parts();
	/* Create storage for observed edges and accessor function */
	let mut observed_edges = ObservedEdgesStore::new(n);
	/* Create an expand queue for edges and insert all edges from all graph levels */
	let mut expand_queue = MinHeap::with_capacity(graphs.iter().map(|g| g.n_edges()).sum::<usize>() + 1_000);
	if !connected {
		/* println!("Warning: HNSW graph is not fully connected!"); */
		/* Add edges between disconnected components and push to expand queue */
		for i in 0..components.len() {
			for j in (i + 1)..components.len() {
				println!("combining components {:?}", (i, j));
				let (s1, s2) = random_sample_points(&components[i], &components[j], 10);

				let new_edges: Vec<(F, (R, R))> = s1
				.par_iter()
				.flat_map(|&i_global| {
					let mut local_edges = Vec::new();
					for &j_global in &s2 {
						if i_global == j_global {
							continue;
						}

						let d = dist.dist_slice(
							data.get_row_view(i_global),
							data.get_row_view(j_global),
						);

						let i_r = unsafe { R::from(i_global).unwrap_unchecked() };
						let j_r = unsafe { R::from(j_global).unwrap_unchecked() };

						local_edges.push((d, (i_r.min(j_r), i_r.max(j_r))));
					}
					let n_keep = 20;
					local_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
					local_edges.truncate(n_keep);
					local_edges
				})
				.collect();

				// Merge results sequentially
				for (d, (i_r, j_r)) in new_edges {
					if !observed_edges.observe_edge(i_r, j_r) {
						expand_queue.push(d, (i_r.min(j_r), i_r.max(j_r)));
					}
				}
			}
		}
	}
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

	let pb = ProgressBar::new((n - 1) as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) | ETA {eta_precise}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );

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
			let new_root = union_find.find(R::from(i_root).unwrap_unchecked()).to_usize().unwrap_unchecked(); // changed from j_root;
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

					/* Attempt to merge with all neighbors first */
					let neighbor_stack = neighbor_stacks.get_unchecked_mut(idx_usize);
					while neighbor_stack.size() > 0 {
						let (distance, other) = neighbor_stack.pop().unwrap_unchecked();
						if *neighbor_counts.get_unchecked(other.to_usize().unwrap_unchecked()) >= min_pts {
							/*` Get root objects */
							let root = union_find.find(idx).to_usize().unwrap_unchecked();
							let other_root = union_find.find(other).to_usize().unwrap_unchecked();
							if root != other_root {
								merge_clusters(
									&mut dendrogram,
									&mut union_find,
									&mut cluster_ids,
									&mut cluster_sizes,
									distance, other_root, root
								);
								pb.inc(1);
							}
						}
					}
				} else if *cnt < min_pts {
					neighbor_stacks.get_unchecked_mut(idx_usize).push(d_ij, other_idx);
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
					d_ij, i_root, j_root,
				);
				pb.inc(1);
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

fn batch_dists<M, F, Dist>(data: &M, dist: &Dist, query: usize, js: &[usize], out: &mut [F])
where
	M: MatrixDataSource<F>,
	F: SyncFloat,
	Dist: Distance<F>,
{
	let qv = data.get_row_view(query);
	for (j, slot) in js.iter().zip(out.iter_mut()) {
		let jv = data.get_row_view(*j);
		*slot = dist.dist_slice(&qv, &jv);
	}
}

// type PrioritySearcherGen<F: SyncFloat> = Gen<(F, usize), (), Pin<Box<dyn Future<Output = ()>>>>;


pub struct PrioritySearcherGenState<F: SyncFloat + Debug + Copy + PartialOrd> {
	expand_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
	candidate_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
	visited: HashSet<usize>,
	peekable: Option<(F, usize)>,
}

impl <F: SyncFloat + Debug + Copy + PartialOrd> PrioritySearcherGenState<F> {
	pub fn new(query: usize) -> Self {
		let mut expand_queue = PriorityQueue::new();
		expand_queue.push(query, Reverse(OrderedFloat(F::zero())));
		let mut visited = foldhash::HashSet::default();
		visited.insert(query);
		Self {
			expand_queue: expand_queue,
			candidate_queue: PriorityQueue::new(),
			visited: visited,
			peekable: Some((F::zero(), query)),
		}
	}
}

// ---- HNSW-HSSL Priority Searcher ---- ////
pub struct PrioritySearcherHnsw<'a, M, F, R, Dist>
where
	M: MatrixDataSource<F> + Sync,
	F: SyncFloat + Debug + Copy + PartialOrd,
	R: SyncUnsignedInteger + Copy,
	Dist: Distance<F> + Sync + Send,
{
	data: &'a M,
	graph: Rc<HNSWHeapBuildGraph<R,F>>,
	dist: Dist,
	query: usize,
	ef: usize,
	// expand_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
	// candidate_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
	// visited: foldhash::HashSet<usize>,
	// union_find: Rc<RefCell<UnionFind<usize>>>, // shared via Rc<RefCell>
	// gen: Option<PrioritySearcherGen<F>>,
	// last: Option<(F, usize)>,
	gen_state: PrioritySearcherGenState<F>,
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
		graph: Rc<HNSWHeapBuildGraph<R,F>>,
		dist: Dist,
		query: usize,
		ef: usize,
		// union_find: Rc<RefCell<UnionFind<usize>>>,
	) -> Self {
		Self {
			data,
			graph,
			dist,
			query,
			ef,
			// expand_queue: PriorityQueue::new(),
			// candidate_queue: PriorityQueue::new(),
			// visited: foldhash::HashSet::default(),
			// union_find,
			// gen: None,  
			// last: None,     
			gen_state: PrioritySearcherGenState::new(query),    
		}
	}

	pub fn advance(&mut self, uf: &mut UnionFind<usize>) -> Option<(F, usize)> {
		/* Just return whatever is pre-computed peekable */
		if self.gen_state.peekable.is_some() {
			let mut ret = None;
			std::mem::swap(&mut self.gen_state.peekable, &mut ret);
			return ret;
		}
		/* Compute new candidates until we have enough */
		while self.gen_state.candidate_queue.len() <= self.ef {
			if self.gen_state.expand_queue.is_empty() { break; }
			/* Get expansion point to populate candidates */
			let (expansion_point, _) = self.gen_state.expand_queue.pop().unwrap();
			/* Get neighbors and filter for unvisited */
			let mut neighbors: Vec<usize> = self.graph
				.neighbors(R::from_usize(expansion_point).unwrap())
				.into_iter()
				.map(|r| r.to_usize().unwrap())
				.filter(|n| !self.gen_state.visited.contains(n))
				.collect();
			/* Reduce list of neighbors to be only points not already in the same cluster */
			let mut reduced_neighbors = Vec::new();
			neighbors.iter().for_each(|&n| {
				if !uf.connected(self.query, n) { reduced_neighbors.push(n); }
				self.gen_state.visited.insert(n);
			});
			neighbors = reduced_neighbors;
			/* Compute distances to possible candidates */
			let mut neighbor_distances = vec![F::zero(); neighbors.len()];
			batch_dists(self.data, &self.dist, self.query, &neighbors, &mut neighbor_distances);
			/* Add possible candidates into state */
			for (&neighbor, &distance) in neighbors.iter().zip(&neighbor_distances) {
				self.gen_state.expand_queue.push(neighbor, Reverse(OrderedFloat(distance)));
				self.gen_state.candidate_queue.push(neighbor, Reverse(OrderedFloat(distance)));
			}
		}
		/* Return whatever is returnable */
		if self.gen_state.candidate_queue.len() > 0 {
			let (idx, Reverse(ord_dist)) = self.gen_state.candidate_queue.pop().unwrap();
			return Some((ord_dist.into_inner(), idx));
		}
		/* End of search, always return None */
		None
	}

	pub fn peek(&mut self, uf: &mut UnionFind<usize>) -> Option<(F, usize)> {
		/* If nothing to peek, try to get something to peek */
		if self.gen_state.peekable.is_none() {
			self.gen_state.peekable = self.advance(uf);
		}
		/* Return whatever is peekable */
		self.gen_state.peekable
	}

}


pub fn hnsw_hssl<'a, F, R, M, Dist>(
	data: &'a M,
	dist: Dist,
	ef: &R,
	hnsw_params: HNSWParams<F>,
) -> (Vec<(usize, usize, F, usize)>, Vec<Duration>)
where
	F: SyncFloat + Debug + Copy + PartialOrd,
	R: SyncUnsignedInteger + Copy,
	M: MatrixDataSource<F> + graphidx::types::Sync,
	Dist: Distance<F> + Sync + Send,
{
	let n = data.n_rows();
	let start_time = Instant::now();

	let mut dendrogram = Vec::with_capacity(n - 1);
	let total_merges = n - 1;
	let milestone_steps = [0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
		.map(|p| ((total_merges as f64) * p).round() as usize);
	let mut milestones = Vec::with_capacity(milestone_steps.len());
	let mut next_milestone_idx = 0;

	/* println!("Building HNSW"); use std::io::*; std::io::stdout().flush(); */
	let mut hnsw = HNSWParallelHeapBuilder::<R, _, _>::base_init(data, dist.clone(), hnsw_params);
	/* println!("HNSW built"); use std::io::*; std::io::stdout().flush(); */
	//let (graphs, _, _global_ids, dist) = hnsw._into_parts();
	let (connected, components) = is_connected_hnsw_full(&hnsw);
	/* println!("Graph is connected: {}", connected); */
	
	let bottom_layer = Rc::new(hnsw.graphs.remove(0));

	let ef_usize = ef.to_usize().unwrap();

	// Shared UnionFind for all searchers
	let mut uf = UnionFind::<usize>::new(n);
	let mut cluster_ids = (0..2*n-1).collect::<Vec<usize>>();

	let mut searchers: Vec<PrioritySearcherHnsw<'a, M, F, R, Dist>> = Vec::with_capacity(n);
	let mut candidate_heap: PriorityQueue<usize, Reverse<OrderedFloat<F>>> = PriorityQueue::new();
	//let dist = Arc::new(dist);

	for i in 0..n {
		let mut searcher = PrioritySearcherHnsw::new(data, bottom_layer.clone(), dist.clone(), i, ef_usize);

		// let gen = searcher.advance_gen(true);
		// searcher.gen = Some(gen);

		// searcher.advance_gen(true);


		//let mut gen = searcher;
		//gen.advance_gen(true);
		//let gen = searcher_clone.borrow_mut();
		
		/*let mut g = PeekableGen {
			searcher,
			gen,
			last: None,
		};*/

		if let Some((dist0, _)) = searcher.peek(&mut uf) {
			candidate_heap.push(i, Reverse(OrderedFloat(dist0)));
		}

		//let searcher_rc = Rc::new(RefCell::new(searcher));

		//let gen = searcher_rc.clone().borrow_mut().advance_gen(true);

		searchers.push(searcher);

		//searchers[0].next();
	}

	let pb = ProgressBar::new(total_merges as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) | ETA {eta_precise}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );

	// Main loop
	while uf.n_subsets() > 1 {
		let (i, Reverse(ord_d)) = candidate_heap.pop().expect("Candidate heap empty");
		let d = ord_d.into_inner();

		let mut next_candidate: Option<(F, usize)> = None;

		let searcher = searchers.get_mut(i).unwrap();
		let searcher_next = searcher.advance(&mut uf);
		if searcher_next.is_some() {
			let (dist2, j) = searcher_next.unwrap();
			if dist2 == d {
				let i_root = uf.find(usize::try_from(i).unwrap());
				let j_root = uf.find(usize::try_from(j).unwrap());
				if i_root != j_root {
					let new_id = n + dendrogram.len();
					dendrogram.push((
						cluster_ids[i_root],
						cluster_ids[j_root],
						num::Float::sqrt(dist2),
						uf.subset_size(i_root) + uf.subset_size(j_root)
					));
					uf.union(i_root, j_root);
					let new_root = uf.find(i_root);
					cluster_ids[new_root] = new_id;

					pb.inc(1);
	
					if next_milestone_idx < milestone_steps.len()
						&& dendrogram.len() >= milestone_steps[next_milestone_idx]
					{
						milestones.push(start_time.elapsed());
						next_milestone_idx += 1;
					}
				}
				if let Some((dist3, _)) = searcher.peek(&mut uf) {
					candidate_heap.push(i, Reverse(OrderedFloat(dist3)));
				}
			} else {
				/* Priority mismatch between searcher-heap and searcher state, just retry */
				candidate_heap.push(i, Reverse(OrderedFloat(dist2)));
			}
		}

		if candidate_heap.is_empty() && uf.n_subsets() > 1 {
			println!("Disconnected components detected, adding shortest bridging edge...");
			use std::io::*;
			std::io::stdout().flush().unwrap();

			let mut n_samples = 50; // number of points to sample per component

			while uf.n_subsets() > 1 {
				let uf_sets = uf.get_sets(); // get current disjoint sets
				let mut min_dist = F::infinity();
				let mut min_pair: Option<(usize, usize)> = None;

				for (i_set_idx, set_i) in uf_sets.iter().enumerate() {
					// sample points from set_i
					let i_points_sample = if set_i.len() <= n_samples {
						set_i.clone()
					} else {
						let mut rng = rand::thread_rng();
						set_i.choose_multiple(&mut rng, n_samples).cloned().collect()
					};

					for set_j in uf_sets.iter().take(i_set_idx) {
						let j_points_sample = if set_j.len() <= n_samples {
							set_j.clone()
						} else {
							let mut rng = rand::thread_rng();
							set_j.choose_multiple(&mut rng, n_samples).cloned().collect()
						};

						for &i_point in &i_points_sample {
							for &j_point in &j_points_sample {
								let distance = dist.dist_slice(
									&data.get_row_view(i_point),
									&data.get_row_view(j_point),
								);

								if distance < min_dist {
									min_dist = distance;
									min_pair = Some((i_point, j_point));
								}
							}
						}
					}
				}

				// merge the components corresponding to the shortest edge
				if let Some((i_point, j_point)) = min_pair {
					let i_root = uf.find(i_point);
					let j_root = uf.find(j_point);

					let new_id = n + dendrogram.len();
					dendrogram.push((
						cluster_ids[i_root],
						cluster_ids[j_root],
						num::Float::sqrt(min_dist),
						uf.subset_size(i_root) + uf.subset_size(j_root),
					));

					uf.union(i_root, j_root);
					let new_root = uf.find(i_root);
					cluster_ids[new_root] = new_id;

					pb.inc(1);

					if next_milestone_idx < milestone_steps.len()
						&& dendrogram.len() >= milestone_steps[next_milestone_idx]
					{
						milestones.push(start_time.elapsed());
						next_milestone_idx += 1;
					}
				} else {
					panic!("Failed to find a bridging edge between disconnected components");
				}
			}
		}

	}

	(dendrogram, milestones)
}

/*
		if candidate_heap.is_empty() && uf.n_subsets() > 1 {
			println!("Oh shoot, I actually have to do it!");
			use std::io::*;
			std::io::stdout().flush();
			let n_sets = uf.n_subsets();
			let n_edges_per_set_pair = 10;
			let uf_sets = uf.get_sets(); /* Expensive! */
			assert!(uf_sets.len() == n_sets);
			assert!(uf_sets.iter().map(|s| s.len()).sum::<usize>() == n);
			/* Sample random edges */
			(0..n_sets).for_each(|i_set| {
				(0..i_set).for_each(|j_set| {
					let (i_points, j_points) = random_sample_points(&uf_sets[i_set], &uf_sets[j_set], n_edges_per_set_pair);
					i_points.into_iter().zip(j_points.into_iter()).for_each(|(i_point, j_point)| {
						let distance = dist.dist_slice(
							&data.get_row_view(i_point),
							&data.get_row_view(j_point),
						);
						searchers[i_point].gen_state.expand_queue.push(j_point, Reverse(OrderedFloat(distance)));
						/* Warning: This might add a searcher **more than once** to the candidate_heap! */
						candidate_heap.push(i_point, Reverse(OrderedFloat(distance))); 
					});
				});
			});
		}
*/

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


// struct PeekableGen<'a, S, F> {
// 	searcher: S,
// 	gen: Gen<(F, usize), (), Pin<Box<dyn Future<Output = ()> + 'a>>>,
// 	last: Option<(F, usize)>,
// }

// impl<'a, S, F> PeekableGen<'a, S, F> {
// 	fn peek(&mut self) -> Option<&(F, usize)> {
// 		if self.last.is_none() {
// 			if let GeneratorState::Yielded(val) = self.gen.resume() {
// 				self.last = Some(val);
// 			}
// 		}
// 		self.last.as_ref()
// 	}

// 	fn next(&mut self) -> Option<(F, usize)> {
// 		if let Some(val) = self.last.take() {
// 			Some(val)
// 		} else if let GeneratorState::Yielded(val) = self.gen.resume() {
// 			Some(val)
// 		} else {
// 			None
// 		}
// 	}
// }

/*fn batch_dists(&self, js: &[usize], out: &mut [F]) {
let qv = self.data.get_row_view(self.query);
for (j, slot) in js.iter().zip(out.iter_mut()) {
	let jv = self.data.get_row_view(*j);
	*slot = self.dist.dist_slice(&qv, &jv); // l2_simd(qv, jv); 
	}
}*/

// pub fn advance_gen(
// 	&'a mut self,
// 	return_self: bool,
// ) {
// 	let data = self.data;
// 	let graph = self.graph.clone();
// 	let dist = self.dist.clone();
// 	let query = self.query;
// 	let ef = self.ef;
// 	let union_find = self.union_find.clone();

// 	let gen = PrioritySearcherGen::new(move |co| {
// 		Box::pin(async move {
// 			if return_self {
// 				co.yield_((F::zero(), query)).await;
// 			}
			
// 			let mut expand_queue = PriorityQueue::new();
// 			let mut candidate_queue = PriorityQueue::new();
// 			let mut visited = foldhash::HashSet::default();

// 			expand_queue.clear();
// 			expand_queue.push(query, Reverse(OrderedFloat(F::zero())));
// 			candidate_queue.clear();
// 			visited.clear();
// 			visited.insert(query);

// 			while let Some((expansion_point, Reverse(ord_dist))) = expand_queue.pop() {
// 				let mut neighbors: Vec<usize> = graph
// 					.neighbors(R::from_usize(expansion_point).unwrap())
// 					.into_iter()
// 					.map(|r| r.to_usize().unwrap())
// 					.collect();

// 				neighbors.retain(|n| !visited.contains(n));

// 				let mut reduced_neighbors = Vec::new();
// 				{
// 					let mut uf = union_find.borrow_mut();
// 					for &n in &neighbors {
// 						if !uf.connected(query, n) {
// 							reduced_neighbors.push(n);
// 						} else {
// 							visited.insert(n);
// 						}
// 					}
// 				}
// 				neighbors = reduced_neighbors;

// 				let mut neighbor_distances = vec![F::zero(); neighbors.len()];
// 				batch_dists(data, &dist, query, &neighbors, &mut neighbor_distances);

// 				for (&neighbor, &distance) in neighbors.iter().zip(&neighbor_distances) {
// 					visited.insert(neighbor);
// 					expand_queue.push(neighbor, Reverse(OrderedFloat(distance)));
// 					candidate_queue.push(neighbor, Reverse(OrderedFloat(distance)));
// 				}

// 				while candidate_queue.len() > ef {
// 					if let Some((idx, Reverse(ord_dist))) = candidate_queue.pop() {
// 						co.yield_((ord_dist.into_inner(), idx)).await;
// 					} else {
// 						break;
// 					}
// 				}
// 			}

// 			while let Some((idx, Reverse(ord_dist))) = candidate_queue.pop() {
// 				co.yield_((ord_dist.into_inner(), idx)).await;
// 			}
// 		}) as Pin<Box<dyn Future<Output = ()> + 'a>>
// 	});
// 	self.gen = Some(gen);
// }

// fn peek(&mut self) -> Option<&(F, usize)> {
// 	if self.last.is_none() {
// 		if let Some(gen) = self.gen.as_mut() {
// 			if let GeneratorState::Yielded(val) = gen.resume() {
// 				self.last = Some(val);
// 			}
// 		}
// 	}
// 	self.last.as_ref()
// }

// fn next(&mut self) -> Option<(F, usize)> {
// 	if let Some(val) = self.last.take() {
// 		Some(val)
// 	} else if let Some(gen) = self.gen.as_mut() {
// 		if let GeneratorState::Yielded(val) = gen.resume() {
// 			Some(val)
// 		} else {
// 			None
// 		}
// 	} else {
// 		None
// 	}
// }