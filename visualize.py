import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import warnings
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from tqdm import tqdm


def plot_losses(train_losses, val_losses):

    sns.set(style="darkgrid", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    plt.title('Loss')
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='val')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig('losses.png')
    plt.show()


def plot_accuracies(train_accs, val_accs):

    sns.set(style="darkgrid", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    plt.title('Accuracy')
    plt.plot(range(1, len(train_accs)+1), train_accs, label='train')
    plt.plot(range(1, len(val_accs)+1), val_accs, label='val')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig('accuracy.png')
    plt.show()


def plot_weights(module_name, layer_name, weights):

    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : 15}
    plt.rc('font', **font)
    axis = 10
    axisplot = 200
    print('\n==> Plotting weights')
    for i in tqdm(range(len(weights))):
        fig, ax = plt.subplots(figsize=(15,10), dpi=350)
        plt.title('Weights Density Distribution \n'+str(layer_name[i])+'\n'+str(module_name[i]))
        ax.set_xlabel('Weight values', color='k')
        ax.set_ylabel('Density', color='b')

        minX, maxX = np.floor(weights[i].min()), np.ceil(weights[i].max())
        bins_plot = np.linspace(minX, maxX, axisplot)
        hist, bins = np.histogram(weights[i], bins=bins_plot)

        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2
        hist[np.abs(center).min()==np.abs(center)]=0
        ax.bar(center, hist, align='center', width=width)

        
        ax.set_xticks(np.arange(minX, maxX , (maxX-minX)/axis) )
        # plt.show()
        plt.savefig('Weight_'+str(i).zfill(2)+'.png', bbox_inches='tight', dpi=300)
        plt.close()


def plot_clustered_distributions(weights, num_clusters=7):

    axis = 10
    axisplot = 200
    num_clusters = [num_clusters]   # стараюсь использовать одинаковое число кластеров для каждого слоя
    for layer in [0,2,4,6,10,20,31,33]:
        print(f'layer #{layer}')
        plt.figure(figsize=(60, 8*len(num_clusters)), dpi=150)
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

        datasets = [(weights[layer], {})]

        plot_num = 1
        for num, k in tqdm(enumerate(num_clusters)):
            print(f'{k} clusters')
            default_base = {'quantile': 0.4,
                            'eps': .1,
                            'damping': .9,
                            'preference': -200,
                            'n_neighbors': 20,
                            'n_clusters': k}

            for index, (dataset, algo_params) in enumerate(datasets):
                
                # update parameters with dataset-specific values
                params = default_base.copy()
                params.update(algo_params)

                X = dataset[dataset!=0][:,None]

                # normalize dataset for easier parameter selection
                X = StandardScaler().fit_transform(X)

                # estimate bandwidth for mean shift
                bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

                # ============
                # Create cluster objects
                # ============
                two_means = cluster.KMeans(n_clusters=params['n_clusters'])
                clustering_algorithms = (('KMeans', two_means),)

                for name, algorithm in clustering_algorithms:
                    t0 = time.time()

                    # catch warnings related to kneighbors_graph
                    with warnings.catch_warnings():
                        algorithm.fit(X)

                    t1 = time.time()
                    if hasattr(algorithm, 'labels_'):
                        y_pred = algorithm.labels_.astype(np.int)
                    else:
                        y_pred = algorithm.predict(X)

                    plt.subplot(len(datasets)*len(num_clusters), len(clustering_algorithms), plot_num)

                    if index == 0:
                        plt.title(name, size=18)
                        plt.ylabel(str(k)+' clusters', color='k', size=18)

                    colors = np.array(list(islice(cycle(['#377eb8']),
                                                  int(max(y_pred) + 1))))

                    minX, maxX = np.floor(X.min()), np.ceil(X.max())
                    bins_plot = np.linspace(minX, maxX, axisplot)
                    hist, bins = np.histogram(X, bins=bins_plot)
                    minY, maxY = hist.min(), hist.max()

                    width = np.diff(bins)
                    center = (bins[:-1] + bins[1:]) / 2
                    hist[np.abs(center).min()==np.abs(center)]=0
                    color = []
                    print(name)
                    for p in range(len(hist)):
                        c = y_pred[np.logical_and(X>bins[p] , X<bins[p+1])[:,0]]
                        if np.size(c)==0:
                            color.append(0)
                        else:
                            color.append( int(np.round(c.mean())))
                    plt.bar(center, hist, align='center', width=width, color=colors[color])
                    plt.xticks(np.arange(minX, maxX , (maxX-minX)/axis) )


                    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                             transform=plt.gca().transAxes, size=15,
                             horizontalalignment='right')
                    plt.text(.99, .01, str(np.unique(y_pred)),
                             transform=plt.gca().transAxes, size=15,
                             horizontalalignment='left')
                    plot_num += 1

        plt.savefig('./Distributions/Clustered'+str(layer).zfill(2)+'.png', bbox_inches='tight', dpi=150)
        plt.close()