import pandas as pd
import matplotlib.pyplot as plt

name = "evaluation/ALOI"

### --- pearson evaluation --- ###
if 1:
    df_hssl = pd.read_csv(name + "hssl_pearson.csv")
    df_kruskal = pd.read_csv(name + "kruskal_pearson.csv")
    df_mst = pd.read_csv(name + "mst_pearson.csv")

    hssl_grouped = df_hssl.groupby('lowest_max_degree')["Pearson Correlation"].max()
    kruskal_grouped = df_kruskal.groupby('lowest_max_degree')["Pearson Correlation"].max()
    mst_grouped = df_mst.groupby('lowest_max_degree')["Pearson Correlation"].max()

    combined_df = pd.DataFrame({
        'HSSL': hssl_grouped,
        'Kruskal': kruskal_grouped,
        'MST': mst_grouped
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    index = list(range(len(combined_df)))

    positions_hssl = [i - bar_width for i in index]
    positions_kruskal = [i for i in index]
    positions_mst = [i + bar_width for i in index]

    ax.bar(positions_hssl, combined_df['HSSL'], bar_width, label='HSSL', alpha=0.25, color="red")
    ax.bar(positions_kruskal, combined_df['Kruskal'], bar_width, label='Kruskal', alpha=0.25, color="blue")
    ax.bar(positions_mst, combined_df['MST'], bar_width, label='MST', alpha=0.25, color="green")

    for i, deg in enumerate(combined_df.index):
        hssl_vals = df_hssl[df_hssl['lowest_max_degree'] == deg]['Pearson Correlation']
        for val in hssl_vals:
            ax.hlines(val, positions_hssl[i] - bar_width/3, positions_hssl[i] + bar_width/3, color='red', linewidth=1)

        kruskal_vals = df_kruskal[df_kruskal['lowest_max_degree'] == deg]['Pearson Correlation']
        for val in kruskal_vals:
            ax.hlines(val, positions_kruskal[i] - bar_width/3, positions_kruskal[i] + bar_width/3, color='blue', linewidth=1)

            mst_vals = df_mst[df_mst['lowest_max_degree'] == deg]['Pearson Correlation']
        for val in mst_vals:
            ax.hlines(val, positions_mst[i] - bar_width/3, positions_mst[i] + bar_width/3, color='green', linewidth=1)

    ax.set_xticks(index)
    ax.set_xticklabels(combined_df.index)
    ax.set_xlabel('Lowest Max Degree')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Quality Comparison by Max Degree')
    ax.legend()
    plt.tight_layout()
    plt.show()


### --- norm evaluation --- ###
if 1:
    df_hssl = pd.read_csv(name + "hssl_norm.csv")
    df_kruskal = pd.read_csv(name + "kruskal_norm.csv")
    df_mst = pd.read_csv(name + "mst_norm.csv")

    hssl_grouped = df_hssl.groupby('lowest_max_degree')["norm"].max()
    kruskal_grouped = df_kruskal.groupby('lowest_max_degree')["norm"].max()
    mst_grouped = df_mst.groupby('lowest_max_degree')["norm"].max()

    combined_df = pd.DataFrame({
        'HSSL': hssl_grouped,
        'Kruskal': kruskal_grouped,
        'MST': mst_grouped
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    index = list(range(len(combined_df)))

    positions_hssl = [i - bar_width for i in index]
    positions_kruskal = [i for i in index]
    positions_mst = [i + bar_width for i in index]

    ax.bar(positions_hssl, combined_df['HSSL'], bar_width, label='HSSL', alpha=0.25, color="red")
    ax.bar(positions_kruskal, combined_df['Kruskal'], bar_width, label='Kruskal', alpha=0.25, color="blue")
    ax.bar(positions_mst, combined_df['MST'], bar_width, label='MST', alpha=0.25, color="green")

    for i, deg in enumerate(combined_df.index):
        hssl_vals = df_hssl[df_hssl['lowest_max_degree'] == deg]['norm']
        for val in hssl_vals:
            ax.hlines(val, positions_hssl[i] - bar_width/3, positions_hssl[i] + bar_width/3, color='red', linewidth=1)

        kruskal_vals = df_kruskal[df_kruskal['lowest_max_degree'] == deg]['norm']
        for val in kruskal_vals:
            ax.hlines(val, positions_kruskal[i] - bar_width/3, positions_kruskal[i] + bar_width/3, color='blue', linewidth=1)

            mst_vals = df_mst[df_mst['lowest_max_degree'] == deg]['norm']
        for val in mst_vals:
            ax.hlines(val, positions_mst[i] - bar_width/3, positions_mst[i] + bar_width/3, color='green', linewidth=1)

    ax.set_xticks(index)
    ax.set_xticklabels(combined_df.index)
    ax.set_xlabel('Lowest Max Degree')
    ax.set_ylabel('norm')
    ax.set_title('Quality Comparison by Max Degree')
    ax.legend()
    plt.tight_layout()
    plt.show()


### mean ratio evaluation --- ###
if 1:
    df_hssl = pd.read_csv(name + "hssl_meanRatio.csv")
    df_kruskal = pd.read_csv(name + "kruskal_meanRatio.csv")
    df_mst = pd.read_csv(name + "mst_meanRatio.csv")

    hssl_grouped = df_hssl.groupby('lowest_max_degree')["mean_ratio"].max()
    kruskal_grouped = df_kruskal.groupby('lowest_max_degree')["mean_ratio"].max()
    mst_grouped = df_mst.groupby('lowest_max_degree')["mean_ratio"].max()

    combined_df = pd.DataFrame({
        'HSSL': hssl_grouped,
        'Kruskal': kruskal_grouped,
        'MST': mst_grouped
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    index = list(range(len(combined_df)))

    positions_hssl = [i - bar_width for i in index]
    positions_kruskal = [i for i in index]
    positions_mst = [i + bar_width for i in index]

    ax.bar(positions_hssl, combined_df['HSSL'], bar_width, label='HSSL', alpha=0.25, color="red")
    ax.bar(positions_kruskal, combined_df['Kruskal'], bar_width, label='Kruskal', alpha=0.25, color="blue")
    ax.bar(positions_mst, combined_df['MST'], bar_width, label='MST', alpha=0.25, color="green")

    for i, deg in enumerate(combined_df.index):
        hssl_vals = df_hssl[df_hssl['lowest_max_degree'] == deg]['mean_ratio']
        for val in hssl_vals:
            ax.hlines(val, positions_hssl[i] - bar_width/3, positions_hssl[i] + bar_width/3, color='red', linewidth=1)

        kruskal_vals = df_kruskal[df_kruskal['lowest_max_degree'] == deg]['mean_ratio']
        for val in kruskal_vals:
            ax.hlines(val, positions_kruskal[i] - bar_width/3, positions_kruskal[i] + bar_width/3, color='blue', linewidth=1)

            mst_vals = df_mst[df_mst['lowest_max_degree'] == deg]['mean_ratio']
        for val in mst_vals:
            ax.hlines(val, positions_mst[i] - bar_width/3, positions_mst[i] + bar_width/3, color='green', linewidth=1)

    ax.set_ylim(0.95, combined_df[['HSSL', 'Kruskal', 'MST']].max().max() * 1.05)

    ax.set_xticks(index)
    ax.set_xticklabels(combined_df.index)
    ax.set_xlabel('Lowest Max Degree')
    ax.set_ylabel('mean_ratio')
    ax.set_title('Quality Comparison by Max Degree')
    ax.legend()
    plt.tight_layout()
    plt.show()


### mean ratio evaluation (averages) --- ###
if 1:
    df_hssl = pd.read_csv(name + "hssl_meanRatio_average.csv")
    df_kruskal = pd.read_csv(name + "kruskal_meanRatio_average.csv")
    df_mst = pd.read_csv(name + "mst_meanRatio_average.csv")

    hssl_grouped = df_hssl.groupby('lowest_max_degree')["mean_ratio"].max()
    kruskal_grouped = df_kruskal.groupby('lowest_max_degree')["mean_ratio"].max()
    mst_grouped = df_mst.groupby('lowest_max_degree')["mean_ratio"].max()

    combined_df = pd.DataFrame({
        'HSSL': hssl_grouped,
        'Kruskal': kruskal_grouped,
        'MST': mst_grouped
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    index = list(range(len(combined_df)))

    positions_hssl = [i - bar_width for i in index]
    positions_kruskal = [i for i in index]
    positions_mst = [i + bar_width for i in index]

    ax.bar(positions_hssl, combined_df['HSSL'], bar_width, label='HSSL', alpha=0.25, color="red")
    ax.bar(positions_kruskal, combined_df['Kruskal'], bar_width, label='Kruskal', alpha=0.25, color="blue")
    ax.bar(positions_mst, combined_df['MST'], bar_width, label='MST', alpha=0.25, color="green")

    for i, deg in enumerate(combined_df.index):
        hssl_vals = df_hssl[df_hssl['lowest_max_degree'] == deg]['mean_ratio']
        for val in hssl_vals:
            ax.hlines(val, positions_hssl[i] - bar_width/3, positions_hssl[i] + bar_width/3, color='red', linewidth=1)

        kruskal_vals = df_kruskal[df_kruskal['lowest_max_degree'] == deg]['mean_ratio']
        for val in kruskal_vals:
            ax.hlines(val, positions_kruskal[i] - bar_width/3, positions_kruskal[i] + bar_width/3, color='blue', linewidth=1)

            mst_vals = df_mst[df_mst['lowest_max_degree'] == deg]['mean_ratio']
        for val in mst_vals:
            ax.hlines(val, positions_mst[i] - bar_width/3, positions_mst[i] + bar_width/3, color='green', linewidth=1)
    
    ax.set_ylim(0.95, combined_df[['HSSL', 'Kruskal', 'MST']].max().max() * 1.05)

    ax.set_xticks(index)
    ax.set_xticklabels(combined_df.index)
    ax.set_xlabel('Lowest Max Degree')
    ax.set_ylabel('mean_ratio')
    ax.set_title('Quality Comparison by Max Degree')
    ax.legend()
    plt.tight_layout()
    plt.show()

