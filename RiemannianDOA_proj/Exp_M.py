# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
from utils import *
from ToolsMC import *

# %%

def exp_M(cohr_flag: bool = False, power_doa_db: np.ndarray = np.array([0, 0, -5]), doa: np.ndarray = np.array([35.25, 43.25, 51.25]),
           basedir:str = '') -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_M_{timestamp}_{str_indp_cohr}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = NUM_MC
    # vec_m = np.array([10, 50, 100, 200])
    vec_m = np.array([10, 100])
    num_configs = len(vec_m)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=vec_m[i], snr= 10 + convert_linear_to_db(vec_m[0]) - convert_linear_to_db(vec_m[i]), N=vec_m[i]*3, 
                power_doa_db=power_doa_db,
                doa=doa,
                cohr_flag=cohr_flag,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$M$', "", vec_m, normalize_rmse_by_parameter=False)
    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'$M$', "", vec_m, normalize_rmse_by_parameter=False)
    # 
    # fig_prob_detection = plot_prob_detection(algos_error_data, r'$M$', "", vec_m)
    # %%
    algo_list = get_algo_dict_list()
    num_configs = len(results)
    num_mc = len(results[0])
    num_algos = len(algo_list)
    num_iters_mat = np.zeros((num_configs, num_mc, num_algos))
    runtime_mat = np.zeros((num_configs, num_mc, num_algos))
    for i_config in range(len(results)):
        num_iters_mat[i_config,:,:] = np.array([results[i_config][i_mc]["num_iters_list"] for i_mc in range(len(results[i_config]))])
        runtime_mat[i_config,:,:] = np.array([results[i_config][i_mc]["runtime_list"] for i_mc in range(len(results[i_config]))])


    algo_names = list(algo_list.keys())
    # --- Plot: Number of Iterations ---
    fig_iters, axes_iters = plt.subplots(1, num_algos, figsize=(4 * num_algos, 4), sharey=True, sharex=True)
    fig_iters.suptitle("Number of Iterations per Configuration")

    for a in range(num_algos):
        data = [num_iters_mat[c, :, a] for c in range(num_configs)]  # list of arrays, each of shape (num_mc,)
        axes_iters[a].boxplot(data, showfliers=False)
        axes_iters[a].set_title(algo_names[a])
        axes_iters[a].set_xlabel("Config Index")
        axes_iters[a].set_xticks(range(1, num_configs + 1))
        if a == 0:
            axes_iters[a].set_ylabel("Num Iters")
        plt.tight_layout()


    # --- Plot: Runtime ---
    fig_runtime, axes_runtime = plt.subplots(1, num_algos, figsize=(4 * num_algos, 4), sharey=True, sharex=True)
    fig_runtime.suptitle("Runtime per Configuration")

    for a in range(num_algos):
        data = [runtime_mat[c, :, a] for c in range(num_configs)]
        axes_runtime[a].boxplot(data, showfliers=False)
        axes_runtime[a].set_title(algo_names[a])
        axes_runtime[a].set_xlabel("Config Index")
        axes_runtime[a].set_xticks(range(1, num_configs + 1))
        if a == 0:
            axes_runtime[a].set_ylabel("Runtime [s]")
        plt.tight_layout()



    iter_runtime_mat = runtime_mat / num_iters_mat  # Shape: (num_configs, num_mc, num_algos)
    fig_iterationruntime, axes_iterationruntime = plt.subplots(1, num_algos, figsize=(4 * num_algos, 4), sharey=True, sharex=True)
    fig_iterationruntime.suptitle("Iteration Runtime per Configuration")
    for a in range(num_algos):
        data = [iter_runtime_mat[c, :, a] for c in range(num_configs)]

        axes_iterationruntime[a].boxplot(data, showfliers=False)
        axes_iterationruntime[a].set_title(algo_names[a])
        axes_iterationruntime[a].set_xlabel("Config Index")
        axes_iterationruntime[a].set_xticks(range(1, num_configs + 1))
        if a == 0:
            axes_iterationruntime[a].set_ylabel("Runtime [s]")
        plt.tight_layout()

    plt.tight_layout()
    #%%
    def create_figure_iterruntime_boxplot(DO_BOXPLOT=True):
            
        # Boxplot settings
        width = 0.15   # width of each box
        x = np.arange(num_configs)  # x locations for each config
        # colors = ['skyblue', 'salmon', 'lightgreen']
        import matplotlib.colors as mcolors

        def lighten_color(color, amount=0.5):
            """
            Lightens the given color by blending it with white.
            amount=0 returns original color, amount=1 returns white.
            """
            try:
                c = mcolors.to_rgb(color)
            except ValueError:
                # if not recognized, default to black
                c = (0,0,0)
            white = np.array([1,1,1])
            return tuple(c + (white - c) * amount)
        original_colors = [algo_list[d]["color"] for d in algo_list]
        colors = [lighten_color(c, 0.5) for c in original_colors] 
        iteration_runtime_ms_mat = iter_runtime_mat * 1e3  # Convert to milliseconds
        fig_runtime_boxplot, ax = plt.subplots(figsize=(6,4))

        for algo_idx in range(num_algos):
            # Shift each algorithm's box a little to the left/right
            spread = 1.1  # 1 = current spacing, >1 = more spread
            pos = x + (algo_idx - num_algos/2 + 0.5) * width * spread
            data_to_plot = [iteration_runtime_ms_mat[config_idx,:,algo_idx] for config_idx in range(num_configs)]
            
            if DO_BOXPLOT:
                bp = ax.boxplot(data_to_plot, positions=pos, widths=width, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[algo_idx])
                for whisker in bp['whiskers']:
                    whisker.set_color('black')
                for cap in bp['caps']:
                    cap.set_color('black')
                for median in bp['medians']:
                    median.set_color('black')
            
            else:
                vp = ax.violinplot(data_to_plot, positions=pos, widths=width, showmeans=False, showmedians=True, showextrema=False)
                #set median color black
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(2)
                # Fill color
                for body in vp['bodies']:
                    body.set_facecolor(colors[algo_idx])
                    body.set_alpha(0.5)  # make it transparent


        # Labels and ticks
        xylabel_fontsize = 12
        ax.set_xticks(x)
        ax.set_xticklabels([f'$M = {vec_m[i]}$' for i in range(num_configs)],fontsize=xylabel_fontsize)
        ax.set_ylabel('Iteration Runtime (ms)',fontsize=xylabel_fontsize)
        # ax.set_title('Algorithm runtimes by configuration')
        ax.set_yscale('log')
        # Legend
        from matplotlib.patches import Patch
        labels = [f"{ALGONAME}({algo_name})" if (algo_name == "AIRM" or algo_name == "JBLD") else algo_name 
                  for algo_name in algo_names]
        legend_handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(num_algos)]
        lgd = ax.legend(handles=legend_handles)
        for text in lgd.get_texts():
            if "JBLD" in text.get_text():
                text.set_fontweight("bold")
        
        delta_x = 0.4
        ax.set_xlim([0-delta_x,1+delta_x])
        plt.tight_layout()
        # plt.show()
        return fig_runtime_boxplot
    
    fig_runtime_boxplot = create_figure_iterruntime_boxplot(DO_BOXPLOT=True)
    fig_runtime_violin = create_figure_iterruntime_boxplot(DO_BOXPLOT=False)


    # %%

    def print_table(table_name, mean_mat, std_mat):
        print("================   " + table_name + "   ================")
        header = ["Config {}".format(i+1) for i in range(num_configs)]
        print("{:<15}".format("Algorithm"), end="")
        for h in header:
            print("{:>20}".format(h), end="")
        print()

        # Print rows
        for alg_idx in range(num_algos):
            print("{:<15}".format(algo_names[alg_idx]), end="")
            for cfg_idx in range(num_configs):
                mean = mean_mat[alg_idx, cfg_idx]
                std = std_mat[alg_idx, cfg_idx]
                print("{:>20}".format(f"{mean:.4f} ± {std:.4f}"), end="")
            print()
        print("==========================================================")

    def print_table_with_percentile(table_name, mat):
        from tabulate import tabulate
        qlow_mat = np.percentile(mat, 10, axis=1).T
        qmid_mat = np.percentile(mat, 50, axis=1).T
        qhigh_mat = np.percentile(mat, 95, axis=1).T

        header = ["Algorithm"] + [f"M={vec_m[i]}" for i in range(num_configs)]
        table_data = []

        for alg_idx in range(num_algos):
            row = [algo_names[alg_idx]]
            for cfg_idx in range(num_configs):
                formatted_value = (
                    f"{qmid_mat[alg_idx, cfg_idx]:.4f} "
                    f"[ {qlow_mat[alg_idx, cfg_idx]:.4f}, "
                    f"{qhigh_mat[alg_idx, cfg_idx]:.4f} ]"
                )
                row.append(formatted_value)
            table_data.append(row)

        print(f"================   {table_name}   ================")
        tabulate_data = tabulate(table_data, headers=header, tablefmt="grid")
        print(tabulate_data)
        print("==========================================================")

        with open(os.path.join(path_results_dir, table_name+'.txt'), "w") as f:
            f.write(tabulate_data)
        return tabulate_data
    
    # print_table("Runtime", runtime_mat.mean(axis=1).T, runtime_mat.std(axis=1).T)
    # print_table("Num Iters", num_iters_mat.mean(axis=1).T, num_iters_mat.std(axis=1).T)
    # print_table("Iteration Runtime", iter_runtime_mat.mean(axis=1).T, iter_runtime_mat.std(axis=1).T)
    # print_table_with_percentile("Runtime", runtime_mat)
    print_table_with_percentile("Num Iters", num_iters_mat)
    print_table_with_percentile("Iteration Runtime [ms]", iter_runtime_mat * 1e3)  # Convert to milliseconds

    
    # %%
    str_desc_name = os.path.basename(name_results_dir)

    # fig_runtime.savefig(os.path.join(path_results_dir, 'Runtime_' + str_desc_name +  '.png'), dpi=300)
    # fig_iters.savefig(os.path.join(path_results_dir, 'IterRuntime_' + str_desc_name +  '.png'), dpi=300)
    # fig_iterationruntime.savefig(os.path.join(path_results_dir, 'NumIters_' + str_desc_name +  '.png'), dpi=300)
    # fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    # fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    # fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)
    save_figure(fig_runtime_boxplot, path_results_dir, str_desc_name+ "_IterRuntimeBoxplot")
    save_figure(fig_runtime_violin, path_results_dir, str_desc_name+ "_IterRuntimeViolin")
    plt.close()
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)

if __name__ == "__main__":
    # Example usage
    exp_M(power_doa_db=np.array([0, 0, 0]), doa=np.array([35.25, 36.25, 37.25]))


# %%
