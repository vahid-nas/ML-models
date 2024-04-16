import numpy as np
from collections import Counter


class Node:
    def __init__(self,feature= None,threshold = None, left=None, right=None, * ,value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split =2, max_depth=5,n_features=None):
        self.min_samples_split =min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.consumed_features = []
        self.cost = None
        self.cost_Instance = None


    def fit(self,X,y,Cost = None):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        if Cost is not None:
            self.cost = Cost
        self.root = self._grow_tree(X,y)


    def _grow_tree(self,X,y,depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_feats,self.n_features,replace = False)

        #find best split
        best_feature,best_threshold = self._best_split(X,y,feat_idx)

        #create child nodes
        left_idxs, right_idxs = self._split(X[:,best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        #print(best_feature,best_threshold,"  ")
        self.consumed_features.append(best_feature)
        #update cost list when a new node is added
        if self.cost is not None:
            self._update_cost()

        return Node(best_feature,best_threshold,left,right)

    def _best_split(self,X,y,feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None,None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate the information gain
                gain = self._information_gain(y,X_column,thr,feat_idx)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def _information_gain(self,y,X_column,threshold,feat_idx):
        #parent enthropy
        parent_entropy = self._entropy(y)


        #create children
        left_idx, right_idx = self._split(X_column,threshold)
        if len(left_idx) == 0 or len(right_idx) == 0 :
            return 0

        #calcluate the weighted average enthropy of children
        n= len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r


        # calculate the information gain
        information_gain = parent_entropy - child_entropy

        if self.cost is None:
            return information_gain
        else:
            self._update_cost()
            return (information_gain * information_gain) / self.cost[feat_idx]




    def _split(self,X_column,split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten() # returns indexes in X_colums that their value is smaller than split_threshold
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self,y):
        hist = np.bincount(y) # returns histogram of the labels
        ps = hist / len(y)  # we calculate the probabilty of every class by this method
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self,y):
        counter = Counter(y)
        value =counter.most_common(1)[0][0]
        return value
    def predict(self,X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self):
        self._trav(self.root)

    def _trav(self,node):
        if node.is_leaf_node():
            print("Leaf"+str(node.value))
        else:
            print("Feature"+str(node.feature) +"   "+str(node.threshold))
            self._trav(node.left)
            self._trav(node.right)

    def print_used_nodes(self):
        print(self.consumed_features)

    def _update_cost(self):
        if np.isin(18, self.consumed_features) and not np.isin(19, self.consumed_features):
            self.cost[20] = self.cost[19]
        elif np.isin(19, self.consumed_features) and not np.isin(18, self.consumed_features):
            self.cost[20] = self.cost[18]
        elif np.isin(19, self.consumed_features) and np.isin(18, self.consumed_features):
            self.cost[20] = 1
        else:
            self.cost[20] = self.cost[18] + self.cost[19]


    def predict_cost(self,X):
        result = []
        for x in X:
            self.cost_Instance = 0
            self._traverse_tree_cost(x,self.root)
            result.append(self.cost_Instance)
        return result
    def _traverse_tree_cost(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            self.cost_Instance += self.cost[node.feature]
            return self._traverse_tree_cost(x,node.left)
        self.cost_Instance += self.cost[node.feature]
        return self._traverse_tree_cost(x, node.right)


