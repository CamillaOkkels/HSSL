import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_pareto_frontier(df, x, y, split):
    # removes all rows that don't lie on the pareto frontier
    to_plot = df.sort_values(x, ascending=True).reset_index(drop=True)
    d = {} # store last x values
    drop_list = []
    for algo in set(df[split]):
        d[algo] = 0
    for i in range(len(to_plot)):
        x_ = to_plot.iloc[i][x]
        y_ = to_plot.iloc[i][y]
        algo = to_plot.iloc[i][split]
        if y_ > d[algo]:
            d[algo] = y_
        else:
            drop_list.append(i)
    to_plot.drop(drop_list, inplace=True)
        
    return to_plot


dataset = "ALOI" # ALOI MNIST
algo = "mst" # hssl, kruskal, mst
measure = "norm" # pearson, norm, mean_ratio
isHSSL = True if algo == "hssl" else False

file_path = dataset + algo + "_" + measure + ".csv"

df_split = pd.read_csv(file_path)

if isHSSL:

    if measure == "pearson":
        df_split = df_split[["algo", "time", "Pearson Correlation", "ef", "max_build_heap_size", "lowest_max_degree"]]
        pareto = get_pareto_frontier(df_split, "time", "Pearson Correlation", "algo")
        pareto[["time", "Pearson Correlation", "ef", "max_build_heap_size", "lowest_max_degree"]]

        filtered_df = pareto[pareto["Pearson Correlation"] > 0.8]
        opt = filtered_df[filtered_df.time == filtered_df.time.min()]
        opt[["time", "Pearson Correlation", "ef", "max_build_heap_size", "lowest_max_degree"]]
        print(opt)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.scatterplot(data=pareto, x="time", y="Pearson Correlation", 
                        hue="ef",
                        style="max_build_heap_size", 
                        size="lowest_max_degree", 
                        palette="colorblind", 
                        # markers={True: "X", False: "o"},
                        sizes=(90, 500))

        plt.title(f"{dataset} - Pearson Correlation over time (hssl)")
        plt.xlabel("Time [s]")
        plt.ylabel("Pearson Correlation")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.xscale("log")

        plt.show()

    if measure == "norm":
        df_split = df_split[["algo", "time", "norm", "ef", "max_build_heap_size", "lowest_max_degree"]]
        pareto = get_pareto_frontier(df_split, "time", "norm", "algo")
        pareto[["time", "norm", "ef", "max_build_heap_size", "lowest_max_degree"]]

        filtered_df = pareto[pareto["norm"] > 0.8]
        opt = filtered_df[filtered_df.time == filtered_df.time.min()]
        opt[["time", "norm", "ef", "max_build_heap_size", "lowest_max_degree"]]
        print(opt)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.scatterplot(data=pareto, x="time", y="norm", 
                        hue="ef",
                        style="max_build_heap_size", 
                        size="lowest_max_degree", 
                        palette="colorblind", 
                        # markers={True: "X", False: "o"},
                        sizes=(90, 500))

        plt.title(f"{dataset} - norm over time (hssl)")
        plt.xlabel("Time [s]")
        plt.ylabel("Norm")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.xscale("log")

        plt.show()

    if measure == "mean_ratio":
        df_split = df_split[["algo", "time", "mean_ratio", "ef", "max_build_heap_size", "lowest_max_degree"]]
        pareto = get_pareto_frontier(df_split, "time", "mean_ratio", "algo")
        pareto[["time", "mean_ratio", "ef", "max_build_heap_size", "lowest_max_degree"]]

        filtered_df = pareto[pareto["mean_ratio"] > 0.8]
        opt = filtered_df[filtered_df.time == filtered_df.time.min()]
        opt[["time", "mean_ratio", "ef", "max_build_heap_size", "lowest_max_degree"]]
        print(opt)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.scatterplot(data=pareto, x="time", y="mean_ratio", 
                        hue="ef",
                        style="max_build_heap_size", 
                        size="lowest_max_degree", 
                        palette="colorblind", 
                        # markers={True: "X", False: "o"},
                        sizes=(90, 500))

        plt.title(f"{dataset} - mean_ratio over time (hssl)")
        plt.xlabel("Time [s]")
        plt.ylabel("Mean_Ratio")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.xscale("log")

        plt.show()

if not isHSSL:

    if measure == "pearson":
        df_split = df_split[["algo", "time", "Pearson Correlation", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]
        pareto = get_pareto_frontier(df_split, "time", "Pearson Correlation", "algo")
        pareto[["time", "Pearson Correlation", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]

        filtered_df = pareto[pareto["Pearson Correlation"] > 0.8]
        opt = filtered_df[filtered_df.time == filtered_df.time.min()]
        opt[["time", "Pearson Correlation", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]
        print(opt)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.scatterplot(data=pareto, x="time", y="Pearson Correlation", 
                        hue="max_build_heap_size", 
                        size="lowest_max_degree", 
                        palette="colorblind", 
                        # markers={True: "X", False: "o"},
                        sizes=(90, 500))

        plt.title(f"{dataset} - Pearson Correlation over time ({algo})")
        plt.xlabel("Time [s]")
        plt.ylabel("Pearson Correlation")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.xscale("log")

        plt.show()

    if measure == "norm":
        df_split = df_split[["algo", "time", "norm", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]
        pareto = get_pareto_frontier(df_split, "time", "norm", "algo")
        pareto[["time", "norm", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]

        filtered_df = pareto[pareto["norm"] > 0.8]
        opt = filtered_df[filtered_df.time == filtered_df.time.min()]
        opt[["time", "norm", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]
        print(opt)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.scatterplot(data=pareto, x="time", y="norm", 
                        hue="max_build_heap_size", 
                        size="lowest_max_degree", 
                        palette="colorblind", 
                        # markers={True: "X", False: "o"},
                        sizes=(90, 500))

        plt.title(f"{dataset} - norm over time ({algo})")
        plt.xlabel("Time [s]")
        plt.ylabel("Norm")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.xscale("log")

        plt.show()

    if measure == "mean_ratio":
        df_split = df_split[["algo", "time", "mean_ratio", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]
        pareto = get_pareto_frontier(df_split, "time", "mean_ratio", "algo")
        pareto[["time", "mean_ratio", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]

        filtered_df = pareto[pareto["mean_ratio"] > 0.8]
        opt = filtered_df[filtered_df.time == filtered_df.time.min()]
        opt[["time", "mean_ratio", "symmetric_expand", "max_build_heap_size", "lowest_max_degree"]]
        print(opt)

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))

        sns.scatterplot(data=pareto, x="time", y="mean_ratio", 
                        hue="max_build_heap_size", 
                        size="lowest_max_degree", 
                        palette="colorblind", 
                        # markers={True: "X", False: "o"},
                        sizes=(90, 500))

        plt.title(f"{dataset} - mean_ratio over time ({algo})")
        plt.xlabel("Time [s]")
        plt.ylabel("Mean_Ratio")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.xscale("log")

        plt.show()
