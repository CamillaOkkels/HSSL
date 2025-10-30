use foldhash::HashSet;
use graphidx::{data::MatrixDataSource, graphs::{Graph, WeightedGraph}, heaps::MinHeap, measures::Distance, types::{SyncFloat, SyncUnsignedInteger}};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::{VecDeque, HashMap};

use crate::hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder};

use rand::Rng;
use std::cmp::Reverse;
use priority_queue::PriorityQueue;
use ordered_float::OrderedFloat;
use std::fmt::Debug;
use num_traits::Float;
use std::time::{Instant, Duration};
use crate::hnsw::HNSWHeapBuildGraph;
use genawaiter::rc::{Gen, Co};
use genawaiter::GeneratorState;
use std::future::Future;
use std::pin::Pin;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

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

fn pick_random_component_and_other(
    components: &Vec<Vec<usize>>,
) -> Option<(usize, usize, usize, usize)> {
    let mut rng = rand::thread_rng();

    if components.len() < 2 {
        return None; // Need at least two components
    }

    // Pick a random component
    let c1_idx = rng.gen_range(0..components.len());
    let c1 = &components[c1_idx];
    if c1.is_empty() {
        return None;
    }

    // Pick a random point from that component
    let p1 = c1[rng.gen_range(0..c1.len())];

    // Pick another component index (from remaining ones)
    let mut other_indices: Vec<usize> =
        (0..components.len()).filter(|&i| i != c1_idx).collect();
    let c2_idx = other_indices[rng.gen_range(0..other_indices.len())];

    let c2 = &components[c2_idx];
    let p2 = c2[rng.gen_range(0..c2.len())];

    Some((c1_idx, p1, c2_idx, p2))
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
    println!("Starting building graph");
    let hnsw = HNSWParallelHeapBuilder::<R,_,_>::base_init(data, dist, hnsw_params);
    println!("Finished building graph");

    /* 
    let (connected, components) = is_connected_hnsw_full(&hnsw);
    println!("Graph is connected: {}", connected);
    // for (idx, comp) in components.iter().enumerate() {
    // println!("Component {} ({} points): {:?}", idx, comp.len(), comp); 
    // }
    if !connected {
        println!("Warning: HNSW graph is not fully connected!");
        let (c1_idx, p1, c2_idx, p2) = pick_random_component_and_other(&components).unwrap();
        println!("{:?}", (c1_idx, p1, c2_idx, p2));

        let closest = hnsw.search_from_point(&data, p1, p2, 100);
    } */



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
                        let (distance, other) = neighbor_stack.pop().unwrap_unchecked();
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


/*
// -------------- HNSW HSSL implementation -------------- //

struct PeekableGen<'a, S, F> {
    searcher: S,
    gen: Gen<(F, usize), (), Pin<Box<dyn Future<Output = ()> + 'a>>>,
    last: Option<(F, usize)>,
}

impl<'a, S, F> PeekableGen<'a, S, F> {
    fn peek(&mut self) -> Option<&(F, usize)> {
        if self.last.is_none() {
            if let GeneratorState::Yielded(val) = self.gen.resume() {
                self.last = Some(val);
            }
        }
        self.last.as_ref()
    }

    fn next(&mut self) -> Option<(F, usize)> {
        if let Some(val) = self.last.take() {
            Some(val)
        } else if let GeneratorState::Yielded(val) = self.gen.resume() {
            Some(val)
        } else {
            None
        }
    }
}

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
    expand_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
    candidate_queue: PriorityQueue<usize, Reverse<OrderedFloat<F>>>,
    visited: foldhash::HashSet<usize>,
    union_find: Rc<RefCell<UnionFind<usize>>>, // shared via Rc<RefCell>
    gen: Option<Gen<(F, usize), (), Pin<Box<dyn Future<Output = ()> + 'a>>>>,
    last: Option<(F, usize)>,
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
        union_find: Rc<RefCell<UnionFind<usize>>>,
    ) -> Self {
        Self {
            data,
            graph,
            dist,
            query,
            ef,
            expand_queue: PriorityQueue::new(),
            candidate_queue: PriorityQueue::new(),
            visited: foldhash::HashSet::default(),
            union_find,
            gen: None,  
            last: None,         
        }
    }


    /*fn batch_dists(&self, js: &[usize], out: &mut [F]) {
    let qv = self.data.get_row_view(self.query);
    for (j, slot) in js.iter().zip(out.iter_mut()) {
        let jv = self.data.get_row_view(*j);
        *slot = self.dist.dist_slice(&qv, &jv); // l2_simd(qv, jv); 
        }
    }*/
    
        pub fn advance_gen(
        &'a self,
        return_self: bool,
    ) -> Gen<(F, usize), (), Pin<Box<dyn std::future::Future<Output = ()> + 'a>>> {
        let data = self.data;
        let graph = self.graph.clone();
        let dist = self.dist.clone();
        let query = self.query;
        let ef = self.ef;
        let union_find = self.union_find.clone();

        Gen::new(move |co| {
            Box::pin(async move {
                if return_self {
                    co.yield_((F::zero(), query)).await;
                }
                
                let mut expand_queue = PriorityQueue::new();
                let mut candidate_queue = PriorityQueue::new();
                let mut visited = foldhash::HashSet::default();

                expand_queue.clear();
                expand_queue.push(query, Reverse(OrderedFloat(F::zero())));
                candidate_queue.clear();
                visited.clear();
                visited.insert(query);

                while let Some((expansion_point, Reverse(ord_dist))) = expand_queue.pop() {
                    let mut neighbors: Vec<usize> = graph
                        .neighbors(R::from_usize(expansion_point).unwrap())
                        .into_iter()
                        .map(|r| r.to_usize().unwrap())
                        .collect();

                    neighbors.retain(|n| !visited.contains(n));

                    let mut reduced_neighbors = Vec::new();
                    {
                        let mut uf = union_find.borrow_mut();
                        for &n in &neighbors {
                            if !uf.connected(query, n) {
                                reduced_neighbors.push(n);
                            } else {
                                visited.insert(n);
                            }
                        }
                    }
                    neighbors = reduced_neighbors;

                    let mut neighbor_distances = vec![F::zero(); neighbors.len()];
                    batch_dists(data, &dist, query, &neighbors, &mut neighbor_distances);

                    for (&neighbor, &distance) in neighbors.iter().zip(&neighbor_distances) {
                        visited.insert(neighbor);
                        expand_queue.push(neighbor, Reverse(OrderedFloat(distance)));
                        candidate_queue.push(neighbor, Reverse(OrderedFloat(distance)));
                    }

                    while candidate_queue.len() > ef {
                        if let Some((idx, Reverse(ord_dist))) = candidate_queue.pop() {
                            co.yield_((ord_dist.into_inner(), idx)).await;
                        } else {
                            break;
                        }
                    }
                }

                while let Some((idx, Reverse(ord_dist))) = candidate_queue.pop() {
                    co.yield_((ord_dist.into_inner(), idx)).await;
                }
            }) as Pin<Box<dyn Future<Output = ()> + 'a>>
        })
    }

    fn peek(&mut self) -> Option<&(F, usize)> {
        if self.last.is_none() {
            if let Some(gen) = self.gen.as_mut() {
                if let GeneratorState::Yielded(val) = gen.resume() {
                    self.last = Some(val);
                }
            }
        }
        self.last.as_ref()
    }

    fn next(&mut self) -> Option<(F, usize)> {
        if let Some(val) = self.last.take() {
            Some(val)
        } else if let Some(gen) = self.gen.as_mut() {
            if let GeneratorState::Yielded(val) = gen.resume() {
                Some(val)
            } else {
                None
            }
        } else {
            None
        }
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

    let dist_for_hnsw = dist.clone();
    let mut hnsw = HNSWParallelHeapBuilder::<R, _, _>::base_init(data, dist, hnsw_params);
    //let (graphs, _, _global_ids, dist) = hnsw._into_parts();
    let bottom_layer = Rc::new(hnsw.graphs.remove(0));

    let ef_usize = ef.to_usize().unwrap();

    // Shared UnionFind for all searchers
    let union_find = Rc::new(RefCell::new(UnionFind::<usize>::new(n)));

    let mut searchers: Vec<PrioritySearcherHnsw<'a, M, F, R, Dist>> = Vec::with_capacity(n);
    let mut candidate_heap: PriorityQueue<usize, Reverse<OrderedFloat<F>>> = PriorityQueue::new();
    //let dist = Arc::new(dist);

    for i in 0..n {
        let uf_ref = union_find.clone();
        let mut searcher = PrioritySearcherHnsw::new(data, bottom_layer.clone(), dist.clone(), i, ef_usize, uf_ref.clone());

        let gen = searcher.advance_gen(true);
        searcher.gen = Some(gen);
        //let mut gen = searcher;
        //gen.advance_gen(true);
        //let gen = searcher_clone.borrow_mut();
        
        /*let mut g = PeekableGen {
            searcher,
            gen,
            last: None,
        };*/

        if let Some(&(dist0, _)) = searcher.peek() {
            candidate_heap.push(i, Reverse(OrderedFloat(dist0)));
        }

        //let searcher_rc = Rc::new(RefCell::new(searcher));

        //let gen = searcher_rc.clone().borrow_mut().advance_gen(true);

        searchers.push(searcher);

        //searchers[0].next();
    }

    // Main loop
    while union_find.borrow().n_subsets() > 1 {
        let (i, Reverse(ord_d)) = candidate_heap.pop().expect("Candidate heap empty");
        let d = ord_d.into_inner();

        let mut next_candidate: Option<(F, usize)> = None;

        {
            let mut g = searchers.get_mut(i).unwrap();
            while let Some((dist2, j)) = g.next() {
                let mut uf = union_find.borrow_mut();
                if dist2 == d && !uf.connected(i, j) {
                    next_candidate = Some((dist2, j));
                    break;
                }
                if dist2 > d {
                    candidate_heap.push(i, Reverse(OrderedFloat(dist2)));
                    break;
                }
            }
        }

        if let Some((dist2, j)) = next_candidate {
            let mut uf = union_find.borrow_mut();
            if !uf.connected(i, j) {
                let i_root = uf.find(usize::try_from(i).unwrap());
                let j_root = uf.find(usize::try_from(j).unwrap());

                dendrogram.push((i_root, j_root, dist2, uf.subset_size(i) + uf.subset_size(j)));
                uf.union(i, j);

                if next_milestone_idx < milestone_steps.len()
                    && dendrogram.len() >= milestone_steps[next_milestone_idx]
                {
                    milestones.push(start_time.elapsed());
                    next_milestone_idx += 1;
                }
            }
        }

        if union_find.borrow().n_subsets() == 1 {
            break;
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

*/