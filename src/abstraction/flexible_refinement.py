import copy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.cluster import HDBSCAN, OPTICS, AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram

import numpy as np

class FlexibleRefinement(object):
    def __init__(self, unstable_state, points, values, _min_samples, _max_clusters, kernel, plot_abstractions=False, filename=""):
        self.unstable_state = unstable_state
        self._points = np.array(points)
        self._values = np.array(values)
        self._min_samples = _min_samples
        self._max_clusters = _max_clusters if len(points) > 20 else 2
        self.kernel = kernel
        self.filename = filename

        both_x_y = False
        if both_x_y:
            self._X = np.hstack((self._points, self._values.reshape(-1, 1)))
        else:
            self._X = [[val] for val in self._values]
            self._X = np.array(self._X)
            self._X.reshape(-1, 1)

        self.clustering = "agglomerative"
        self.classifier = "svm"

        self.threshold = 0.1

        self.plot_abstractions = plot_abstractions
        self.plot_decision_boundaries = True if len(unstable_state.state) == 2 else False
        if self.plot_abstractions:
            self.plt = plt.figure(figsize=(9, 12))
            G = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 2], width_ratios=[1, 1], wspace=0.1, hspace=0.2)
            self.ax1 = plt.subplot(G[0, 0])
            self.ax2 = plt.subplot(G[0, 1])
            self.ax3 = plt.subplot(G[1, :2])
            self.ax4 = plt.subplot(G[2, 0])
            self.ax5 = plt.subplot(G[2, 1])
            self.space = np.arange(len(self._points))
            self.plt, self.ax1 = self.plot_values(self.plt, self.ax1)
        
    ######################## find_clusters ########################

    def find_clusters(self):  
        reachability_ordered = None 
        labels_ordered = None     
        if self.clustering == "optics":
            db = OPTICS(min_samples=self._min_samples, xi=0.05, min_cluster_size=self._min_samples).fit(self._X)
            reachability_ordered, labels_ordered = db.reachability_[db.ordering_], db.labels_[db.ordering_]
        elif self.clustering == "agglomerative":
            n_clusters = 1000
            distance_threshold = 0.1
            while n_clusters > self._max_clusters:
                db = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None, linkage="ward").fit(self._X)
                n_clusters = len(np.unique(db.labels_))
                distance_threshold += 0.01
            # self.plot_dendrogram(db, self._X)
        elif self.clustering == "hdbscan":
            db = HDBSCAN(min_samples=self._min_samples, min_cluster_size=self._min_samples).fit(self._X)
  
        labels = db.labels_
        if labels_ordered is None:
            labels_ordered = labels
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_clusters = len(np.unique(labels))
        if self.plot_abstractions:
            self.ax2, self.ax3 = self.plot_clusters_and_reachability(reachability_ordered, labels_ordered, self.ax2, self.ax3)

        clf = None
        y_pred = []
        if num_clusters > 1:
            clf, kernel, y_pred = self.find_decision_boundaries(self._points, labels)
            if self.plot_decision_boundaries:
                self.ax4, handles1, custom_labels1 = self.plot_decision_boundaries(clf, kernel, self._points, y_pred, self.ax4)
                first_legend = self.ax4.legend(handles1, custom_labels1, loc= "upper left", bbox_to_anchor=(-0.3, 0.78), title="Classes1")
                self.ax4.add_artist(first_legend)

            if self.classifier == "svm":
                labels = self.merge_clusters(labels, self.threshold, self._max_clusters)
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                clf, kernel, y_pred = self.find_decision_boundaries(self._points, labels)
                if self.plot_decision_boundaries:
                    self.ax5, handles2, custom_labels2 = self.plot_decision_boundaries(clf, kernel, self._points, y_pred, self.ax5)
                    first_legend = self.ax4.legend(handles1, custom_labels1, loc= "upper left", bbox_to_anchor=(-1.5, 0.78), title="Classes1")
                    self.ax4.add_artist(first_legend)
                    second_legend = self.ax5.legend(handles2, custom_labels2, loc= "upper right", bbox_to_anchor=(1.28, 0.78), title="Classes2")
                    self.ax5.add_artist(second_legend)

            if clf is not None and len(set(y_pred)) < 2:
                clf = None

        if self.plot_abstractions:        
            self.plt.suptitle(f"Clustering and decision boundaries for ({str(self.unstable_state)})")
            self.plt.savefig(self.filename)
            plt.close('all')
        return n_clusters, labels, clf, y_pred
     
    ###################### merge clusters ######################

    def merge_clusters(self, labels, merge_threshold, max_clusters):
        current_labels = self.assign_noise_points_to_nearest_cluster(labels, merge_threshold)
        num_clusters = len(np.unique(current_labels))

        merge_done = True
        while merge_done:
            merge_done = False

            # Calculate centroids for the current clusters
            unique_labels = np.unique(current_labels)
            labels_to_centroids = self.compute_centroids(current_labels)
            centroids = np.array([labels_to_centroids[label] for label in unique_labels])

            # Ensure centroids are a 2D array
            if centroids.ndim == 1:
                centroids = centroids.reshape(-1, 1)  # Convert to 2D if it's 1D

            # Calculate pairwise distances between centroids
            distances = cdist(centroids, centroids)

            # Set diagonal to infinity to ignore self-pairing
            np.fill_diagonal(distances, np.inf)

            # Find the pair of clusters with the smallest distance
            min_dist = np.min(distances)
            if min_dist >= merge_threshold and len(unique_labels) <= max_clusters:
                break
            
            idx1, idx2 = np.unravel_index(np.argmin(distances), distances.shape)
            label1, label2 = unique_labels[idx1], unique_labels[idx2]

            min_label = min(label1, label2)
            max_label = max(label1, label2)

            current_labels[current_labels == max_label] = min_label
            merge_done = True
        
        labels = self.relabel_clusters(current_labels)
        return labels

    def compute_centroids(self, current_labels):
        unique_labels = np.unique(current_labels)
        labels_to_centroids = {}
        for label in unique_labels:
            labels_to_centroids[label] = self._values[current_labels == label].mean(axis=0)
        return labels_to_centroids

    def assign_noise_points_to_nearest_cluster(self, labels, threshold):
        # Assign noise points
        labels_to_centroids = self.compute_centroids(labels)
        for i in range(len(labels)):
            if labels[i] == -1:
                distances = []
                for label, mean in labels_to_centroids.items():
                    distance = np.linalg.norm(self._values[i] - mean)
                    distances.append((distance, label))
                if distances:
                    min_distance, nearest_cluster = min(distances)
                    if min_distance < threshold:
                        labels[i] = nearest_cluster
                    else:
                        new_label = max(labels) + 1
                        labels[i] = new_label
                        labels_to_centroids[new_label] = self._values[i]
                    labels_to_centroids = self.compute_centroids(labels)
        return labels

    # Relabel clusters to start from 0
    def relabel_clusters(self, labels):
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        new_labels = np.array([label_mapping[label] for label in labels])
        return new_labels

    ####################### find decision boundaries ########################

    def find_decision_boundaries(self, X, y):
        print("Finding decision boundaries")
        clf = None
        # kernel = None
        if self.classifier == "svm":
            # for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            classes = np.unique(y)
            weight_list = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            weight = {classes[i]:weight_list[i] for i in range(len((weight_list)))}

            # param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
            param_grid = {'C': [0.1, 1, 10]}
            # Determine the number of samples in the smallest class
            from collections import Counter
            class_counts = Counter(y)
            min_class_count = min(class_counts.values())
            n_splits = min(5, min_class_count)
            if n_splits >= 2:
                grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=n_splits)
                grid_search.fit(X, y)
                best_C = grid_search.best_params_['C']
                # best_gamma = grid_search.best_params_['gamma']
                best_gamma = "scale"
            else:
                best_C = 1.0 # Higher C → Less regularization (fits data more closely, but can overfit)
                             # Lower C → More regularization (simplifies the model, prevents overfitting, but may underfit)
                best_gamma = "scale" # prevents too high gamma (overfitting) or too low gamma (underfitting)

            svc = svm.SVC(kernel=self.kernel, 
                        C=best_C,
                        gamma=best_gamma, 
                        class_weight=weight, 
                        max_iter=3000 
                    )
            clf = svc.fit(X, y)
            y_pred = clf.predict(X)
        elif self.classifier == "nn":
            clf = MLPClassifier(hidden_layer_sizes=(128, 128, 64, 16),
                                activation='relu',
                                solver='adam', 
                                alpha=1e-4,
                                learning_rate = 'adaptive', # Only used when solver='sgd', else 'constant'
                                learning_rate_init=0.01,
                                max_iter=100000,
                                random_state=1,
                                epsilon=1e-8,
                                momentum=0.9, # Only used when solver='sgd', else 0.0
                                early_stopping = False,
                                n_iter_no_change = 200,
                                verbose=True)
            clf.fit(X, y)
            y_pred = clf.predict(X)
        return clf, self.kernel, y_pred

    ########################### plotting functions ###########################

    def plot_values(self, plt, ax):
        scatter = ax.scatter(self._points[:, 0], self._points[:, 1], c=list(self._values), s=15, cmap="viridis")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Value')
        plt.suptitle(f"Clustering and decision boundaries for ({str(self.unstable_state)})")
        plt.savefig(self.filename)
        return plt, ax

    def plot_clusters_and_reachability(self, reachability, labels, ax1, ax2):
        print("Plotting clusters and reachability")
        colors_map = {}
        for i, label in enumerate(np.unique(labels)):
            if label == -1:
                colors_map[label] = 'k'
            else:
                colors_map[label] = f'C{i}'
        colors = [colors_map[label] for label in labels]

        scatter = ax1.scatter(self._points[:, 0], self._points[:, 1], c=colors, s=15)
        ax1.set_title(f"Automatic Clustering {self.clustering}")

        if reachability is not None:
            for label in np.unique(labels):
                ax2.plot(self.space[labels == label], reachability[labels == label], color=colors[label], markersize=4)
            ax2.set_ylabel("Reachability (epsilon distance)")
            ax2.set_xlabel("Sample index")
            ax2.set_title("Reachability Plot")
        return ax1, ax2
    
    def plot_dendrogram(self, model, X):
        # Extract children and distances
        children = model.children_
        distances = model.distances_

        # Create the linkage matrix
        n_samples = len(X)
        linkage_matrix = np.column_stack([
            children,
            distances,
            np.arange(2, n_samples + 1)
        ]).astype(float)

        # Extract minimum and maximum distances
        min_distance = distances.min()
        max_distance = distances.max()

        print(f"Minimum distance: {min_distance}")
        print(f"Maximum distance: {max_distance}")

        dendrogram(linkage_matrix)
        # plt.savefig(self.filename.split(".png")[0] + "_dendrogram.png")
    
    def plot_decision_boundaries(self, clf, kernel, X, decisions, ax):
        print("Plotting decision boundaries")
        X = np.array(X)
        x_min, x_max, y_min, y_max = min(X[:,0]), max(X[:,0])+0.01, min(X[:,1]), max(X[:,1])+0.01
        # _, ax = plt.subplots(figsize=(6,6))
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

        if clf is not None:
            # Plot decision boundary and margins
            common_params = {"estimator": clf, "X": X, "ax": ax}
            DecisionBoundaryDisplay.from_estimator(
                **common_params,
                response_method="predict",
                plot_method="pcolormesh",
                alpha=0.3,
            )
            DecisionBoundaryDisplay.from_estimator(
                **common_params,
                response_method="predict", #"decision_function",
                plot_method="contour",
                levels=[-1, 0, 1],
                colors=["k", "k", "k"],
                linestyles=["--", "-", "--"],
            )

        class_to_qvalues = dict()
        for i in range(len(decisions)):
            if decisions[i] not in class_to_qvalues:
                class_to_qvalues[decisions[i]] = [self._values[i]]
            else:
                class_to_qvalues[decisions[i]].append(self._values[i])

        class_to_mean_qvalue = dict()
        for _class, qvalues in class_to_qvalues.items():
            mean_qvalue = np.mean(qvalues)
            class_to_mean_qvalue[_class] = mean_qvalue

        # Plot samples by color and add legend
        scatter = ax.scatter(X[:, 0], X[:, 1], c=decisions, s=15, edgecolors="k")
        handles, labels = scatter.legend_elements(prop="colors")
        labels = [int(_class.strip("$\\mathdefault{}$")) for _class in labels]
        custom_labels = [f"{_class}_{round(class_to_mean_qvalue[_class],2)}_{len(class_to_qvalues[_class])}" for _class in labels]
        if clf is not None:
            ax.set_title(f" Decision boundaries using {kernel} kernel")
        else:
            ax.set_title(f" Clusters")
        return ax, handles, custom_labels

