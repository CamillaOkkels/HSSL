import pandas as pd
from HSSL import *
# %% #### Imports ####
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from benchmark.results import load_all_results

def find_cut_old(dendrogram):
	distances = [d[2] for d in dendrogram]
	distances.sort(reverse=True)
	index = 0
	max_jump = -1
	for i in range(len(distances) - 1):
		d = (distances[i] - distances[i + 1]) / distances[i]
		if d > max_jump:
			max_jump = d
			index = i
	return index + 2
def find_cut_old2(dendrogram, t=1.5, criterion='inconsistent'):
	from scipy.cluster.hierarchy import fcluster
	clustering = fcluster(dendrogram, t=t, criterion=criterion)
	# print(clustering)
	return len(np.unique(clustering[clustering>=0]))
def find_cut(dendrogram):
	# https://github.com/sharpenb/Hierarchical-Paris-Clustering/
	from python_paris.homogeneous_cut_slicer import best_homogeneous_cut
	cut_level, cut_score = best_homogeneous_cut(np.array(dendrogram))
	# print(cut_level, cut_score)
	return len(dendrogram) + 1 - cut_level
def fosc_from_dendrogram(dendrogram):
	from FOSC import FOSC
	# Default value for mClSize: 4
	foscFramework = FOSC(dendrogram=dendrogram, mClSize=4)
	infiniteStability = foscFramework.propagateTree()
	partition, lastObjects = foscFramework.findProminentClusters(1, infiniteStability)
	return partition

def get_clustering_from_dendrogram(dendrogram, k, min_cluster_size=10):
	# roundup_fix_dendrogram(dendrogram)
	N = len(dendrogram) + 1
	clustering = -np.ones(N+len(dendrogram), dtype=int)
	merge_distance_order = np.argsort([v[2] for v in dendrogram], kind="stable")
	final_clusters = set([2*N-2])
	i = len(merge_distance_order)-1
	while len(final_clusters) < k:
		final_clusters.remove(N+i)
		l,r = dendrogram[merge_distance_order[i]][:2]
		l_size = 1 if l<N else dendrogram[l-N][3]
		r_size = 1 if r<N else dendrogram[r-N][3]
		if l_size >= min_cluster_size: final_clusters.add(l)
		if r_size >= min_cluster_size: final_clusters.add(r)
		i -= 1
	final_clusters = np.sort([*final_clusters])
	for i,c in enumerate(final_clusters): clustering[c] = i
	for merged_index_diff, (cluster_i, cluster_j, _, _) in enumerate(reversed(dendrogram)):
		merged_index = N + len(dendrogram) - merged_index_diff - 1
		if merged_index > final_clusters[-1]: continue
		clustering[[cluster_i, cluster_j]] = clustering[merged_index]
	return clustering[:N]

def ARI_score(dendro1, dendro2): 
	dendro1 = [[int(l), int(r), float(d), int(s)] for l, r, d, s in dendro1]
	dendro2 = [[int(l), int(r), float(d), int(s)] for l, r, d, s in dendro2]

	dendro1 = elki_sort_dendrogram(dendro1)
	dendro2 = elki_sort_dendrogram(dendro2)
	
	k = find_cut(dendro1)

	C1 = get_clustering_from_dendrogram(dendro1, k)
	C2 = get_clustering_from_dendrogram(dendro2, k)

	return adjusted_rand_score(C1, C2)


if 0: # Compare VPTree and HNSW dendrograms
	path = r'results\blobs-32k-10-5\VPTreehssl\run.hdf5'
	with h5py.File(path, 'r') as f:
		print(list(f.keys()))
		den_sl = f['dendrogram'][:]

	den_sl = [[int(a), int(b), c, int(d)] for a, b, c, d in den_sl]
	# plotly_dendrogram(den_sl, min_size=2, largest_left=False)

	path = r'results\blobs-32k-10-5\HNSWhssl\20_200_20.hdf5'
	with h5py.File(path, 'r') as f:
		print(list(f.keys()))
		den_hnsw = f['dendrogram'][:]

	den_hnsw = [[int(a), int(b), c, int(d)] for a, b, c, d in den_hnsw]
	# plotly_dendrogram(den_hnsw, min_size=2, largest_left=False)

	print(find_cut(den_sl), find_cut(den_hnsw))

	print("ARI:",ARI_score(den_sl, den_hnsw))

	print(np.sort([v[2] for v in den_sl])[-10:])
	print(np.sort([v[2] for v in den_hnsw])[-10:])

	if 0: # Use FOSC?
		fosc_sl = fosc_from_dendrogram(den_sl)
		fosc_hnsw = fosc_from_dendrogram(den_hnsw)
		print(np.unique(fosc_sl, return_counts=True))
		print(fosc_sl)
		print(np.unique(fosc_hnsw, return_counts=True))
		print(fosc_hnsw)


if 1: # Load data for some benchmark and visualize results
	if 1: # Load data
		datasets = ['blobs-32k-10-5']
		data = []
		exact_data = {}
		# load all the results that are available for the dataset
		for dataset in datasets:
			for f in load_all_results(dataset, ""):
				if f.attrs['algo'] == 'VPTreehssl':
					exact_data[dataset] = f["dendrogram"][:]
					break
		for dataset in datasets:
			for f in load_all_results(dataset, ""):
				if f.attrs['algo'] == "HNSWhssl":
					ARI = ARI_score(exact_data[dataset], f["dendrogram"][:])
					data.append({
						"algo": f.attrs['algo'],
						"time": f.attrs['time'],
						"n": len(f["dendrogram"][:]) + 1,
						"params": f.attrs["params"],
						"ARI": ARI
					})
		df = pd.DataFrame(data=data)

	if 1: # Add missing columns if necessary
		import json

		if any(col not in df.columns for col in ['M', 'ef_construct', 'ef']):
			df["params_dict"] = df["params"].apply(lambda x: json.loads(x))
			params_df = df["params_dict"].apply(pd.Series)
			df = pd.concat([df, params_df], axis=1)

	if 0: # Visualize results (seaborn, ugh)
		import seaborn as sns

		sns.set(style="darkgrid")
		plt.figure(figsize=(10, 6))
		sns.scatterplot(data=df, x="time", y="ARI", hue="ef_construct", style="M", palette="colorblind", sizes=(50, 200))

		fixed_M = 20
		fixed_M2 = 10
		fixed_ef_construct = 50

		lines1_df = df[df['M'] == fixed_M]
		lines1_df2 = df[df['M'] == fixed_M2]
		lines2_df = df[df['ef_construct'] == fixed_ef_construct]

		sns.lineplot(data=lines1_df, x='time', y='ARI', linestyle='--', label=f'M = {fixed_M}')
		sns.lineplot(data=lines1_df2, x='time', y='ARI', linestyle='-.', label=f'M = {fixed_M2}')
		sns.lineplot(data=lines2_df, x='time', y='ARI', linestyle=':', label=f'ef_construct = {fixed_ef_construct}')

		plt.title("Blobs 32k - ARI over time")
		plt.xlabel("Time [s]")
		plt.ylabel("ARI score")
		plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
		plt.tight_layout()
		plt.show()
	if 1: # Visulize results (plotly, nice)
		import plotly.graph_objects as go
		import ethcolor as ec

		def ordered_select(target_array, input_values, lookup_table=None):
			if lookup_table is None: lookup_table = input_values
			unique_lookup = np.unique(lookup_table)
			inverse_lookup = {v: i for i, v in enumerate(unique_lookup)}
			index = np.array([inverse_lookup[v] for v in input_values])
			return target_array[index]
		def sort_xy_by(x,y,sort_by):
			order = np.argsort(sort_by, kind="stable")
			return dict(x=x[order], y=y[order])

		times = df["time"].to_numpy()
		aris = df["ARI"].to_numpy()
		ms = df["M"].to_numpy()
		ef_constructs = df["ef_construct"].to_numpy()
		
		order = np.argsort(ef_constructs[np.argsort(ms, kind="stable")], kind="stable")
		times,aris,ms,ef_constructs = times[order], aris[order], ms[order], ef_constructs[order]

		mark_symbols = np.array(["circle", "cross", "x", "diamond", "square"])
		colors = np.array([ec.default_palettes.get_palette().get_color(i).get_value(ec.COLOR_FORMATS.rgb_S) for i in range(1,6)])

		fixed_M = 20
		fixed_M2 = 10
		fixed_ef_construct = 50

		fig = go.Figure()
		for i, ef_construct in enumerate(np.unique(ef_constructs)):
			fig.add_trace(go.Scatter(
				x=[None],y=[None],mode='markers',
				marker_symbol="circle",
				marker_color=ordered_select(colors, [ef_construct], ef_constructs)[0],
				name=f"ef_construct = {ef_construct}",
				showlegend=True,
			))
		for i, m in enumerate(np.unique(ms)):
			fig.add_trace(go.Scatter(
				x=[None],y=[None],mode='markers',
				marker_symbol=ordered_select(mark_symbols, [m], ms)[0],
				marker_color="black",
				name=f"M = {m}",
				showlegend=True,
			))
		fig.add_trace(go.Scatter(
			**sort_xy_by(
				x=times[ms == fixed_M],
				y=aris[ms == fixed_M],
				sort_by=ef_constructs[ms == fixed_M],
			),
			mode='lines',
			marker_symbol=ordered_select(mark_symbols, [fixed_M], ms)[0],
			marker_color=ordered_select(colors, ef_constructs),
			line_dash="dash",
			line_color="black",
			name=f"M = {fixed_M}",
			showlegend=True,
		))
		fig.add_trace(go.Scatter(
			**sort_xy_by(
				x=times[ms == fixed_M2],
				y=aris[ms == fixed_M2],
				sort_by=ef_constructs[ms == fixed_M2],
			),
			mode='lines',
			marker_symbol=ordered_select(mark_symbols, [fixed_M2], ms)[0],
			marker_color=ordered_select(colors, ef_constructs),
			line_dash="dashdot",
			line_color="black",
			name=f"M = {fixed_M2}",
			showlegend=True,
		))
		fig.add_trace(go.Scatter(
			**sort_xy_by(
				x=times[ef_constructs == fixed_ef_construct],
				y=aris[ef_constructs == fixed_ef_construct],
				sort_by=ms[ef_constructs == fixed_ef_construct],
			),
			mode='lines',
			marker_symbol=ordered_select(mark_symbols, ms),
			marker_color=ordered_select(colors, [fixed_ef_construct], ef_constructs)[0],
			line_dash="dot",
			line_color="black",
			name=f"ef_construct = {fixed_ef_construct}",
			showlegend=True,
		))
		fig.add_trace(go.Scatter(
			x=times,
			y=aris,
			mode='markers',
			marker_symbol=ordered_select(mark_symbols, ms),
			marker_color=ordered_select(colors, ef_constructs),
			name="General data",
			showlegend=False,
		))
		fig.update_layout(
			title="Blobs 32k - ARI over time",
			xaxis_title="Time [s]",
			yaxis_title="ARI score",
		)
		fig.show()
		# fig.show(renderer="browser")


		from plotly import express as ex
		color_map = {k:v for k,v in zip(np.sort(np.unique(ef_constructs)), colors)}
		go.Figure(ex.scatter(
			data_frame=df,
			x="time",
			y="ARI",
			color="ef_construct",
			symbol="M",
			color_discrete_sequence=[color_map[k] for k in df["ef_construct"]],
			# color_discrete_map=color_map,
			color_continuous_scale=None,
		)).show()





