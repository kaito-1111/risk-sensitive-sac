import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
risk_coef_list = [-0.03,-0.02,-0.01,0.01,0.02,0.03]
ent_coef_list = [0.01,0.1,1.0]
l_list = [1.0,1.25,1.5]

plt.rc("text",usetex=True)

def figure_robustness():
    l_label = [r"$l={}$".format(l) for l in l_list]
    eta_label = [r"$\eta={}$".format(risk_coef) for risk_coef in risk_coef_list]
    eta_label = eta_label[1:5]
    eta_label.insert(2,"SAC")
    prefix = './data/'
    data_set = []
    for l in l_list:
        data_tmp = np.loadtxt(prefix+"robustness_dataset_l{}.csv".format(l))
        data_set.append(-data_tmp)

    data = {
    l_label[i]: (data_set[i].mean(axis=0),
                 data_set[i].max(axis=0)-data_set[i].mean(axis=0),
                -data_set[i].min(axis=0)+data_set[i].mean(axis=0))
                for i in range(len(l_list))
            }

    x = np.arange(len(eta_label))  # the label locations
    width = 0.1  # the width of the bars
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in data.items():
        ax.errorbar(x, measurement[0], yerr=(measurement[2],measurement[1]),fmt="-o",
        capsize=4,capthick=3,elinewidth=3, label=attribute, markeredgecolor="black")
       

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average episode cost',fontsize = 14)
    ax.set_xticks(x + width, eta_label,fontsize =14)
    ax.legend(loc='upper left', ncols=3,fontsize=14)
    ax.set_ylim(100, 850)
    #plt.grid()
    plt.savefig("./plot/robustness.pdf")
    plt.show()

def figure_distribution_pendulum():
    l_label = [r"$l={}$".format(l) for l in l_list]
    eta_label = [r"$\eta={}$".format(risk_coef) for risk_coef in risk_coef_list]
    eta_label = eta_label[1:5]
    eta_label.insert(2,"SAC")
    prefix = './data/'
    
    
    
    for idx, l in enumerate(l_list):
        fig, axs = plt.subplots(1, 1, figsize=(8, 5), layout='constrained')
        data_set = -np.loadtxt(prefix + "distribution_dataset_l{}.csv".format(l))
        if  type(axs) is list:
            ax = axs[idx]
        else:
            ax = axs
        
        for i in range(5):
            data = data_set[i, :]
            kde = gaussian_kde(data)
            x_kde = np.linspace(min(data) - 1, max(data) + 1, 1000)
            ax.plot(x_kde, kde(x_kde), linewidth=2, label=eta_label[i])
        
        ax.set_title(l_label[idx], fontsize=14)
        ax.set_xlabel('Episode cost', fontsize=14)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=14)
        ax.legend(loc='upper right', ncols=3, fontsize=10)
        ax.set_xlim([0,1200])
        ax.set_ylim([0,0.012])
        plt.savefig("./plot/distribution_penduluml{}.pdf".format(l))
        plt.show()

