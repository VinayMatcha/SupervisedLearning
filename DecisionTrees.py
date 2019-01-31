# import numpy as np
# from Util import get_data, get_donut, get_xor
#
#
# def entropy(Y):
#     N = len(Y)
#     s1 = (Y==1).sum()
#     if s1 == 0 or s1 == N:
#         return 0
#     p1 = float(s1)/N
#     p0 = 1 - p1
#     return -p0 * np.log2(p0) - p1 * np.log2(p1)
#
#
# class TreeNode:
#
#     def __init__(self, depth=0, max_depth  = None):
#         self.depth = depth
#         self.max_depth = max_depth
#
#     def fit(self, X, Y):
#         if(len(Y) == 1 or len(set(Y)) == 1):
#             self.col = None
#             self.split = None
#             self.left = None
#             self.right = None
#             self.prediction = Y[0]
#         else:
#             N, D = X.shape
#             max_ig = 0
#             best_col = None
#             best_split = None
#             for col in range(D):
#                 ig, split = self.find_split(X, Y, col)
#                 if(ig > max_ig):
#                     max_ig = ig
#                     best_col = col
#                     best_split = split
#
#             if max_ig == 0:
#                 self.col = None
#                 self.split = None
#                 self.left = None
#                 self.right = None
#                 self.prediction = np.round(Y.mean())
#             else:
#                 self.col = best_col
#                 self.split = best_split
#
#                 if self.split == self.max_depth:
#                     self.left = None
#                     self.right = None
#                     self.prediction = [
#                         np.round(Y[X[:, best_col] < self.split].mean())
#                         np.round(Y[X[:, best_col] >= self.split].mean())
#                     ]
#                 else:
#                     left_idx = (X[:, best_col] < best_split)
#                     Xleft = X[left_idx]
#                     Yleft = Y[left_idx]
#                     self.left = TreeNode(self.depth+1, self.max_depth)
#                     self.left.fit(Xleft, Yleft)
#
#                     right_idx = (X[:, best_col] >= best_split)
#                     Xright = X[right_idx]
#                     Yright = Y[right_idx]
#                     self.right = TreeNode(self.depth + 1, self.max_depth)
#                     self.right.fit(Xright, Yright)
#
#     def find_split(self, X, Y, col):
#         x_values = X[:, col]
#         # sorting the args depending on the values
#         sort_idx = np.argsort(x_values)
#         x_values = X[sort_idx]
#         y_values = Y[sort_idx]
#
#         boundaries = np.nonzero(y_values[:-1] != y_values)
#         best_split = None
#         max_ig = 0
#         for i in range(len(boundaries)):
#             split = (x_values[i] * x_values[i+1])/2
#
#
#
