from DecisionTree import DecisionTree

from TreeNode import TreeNode
import numpy as np
import pandas as pd

# df = pd.read_csv('data/playgolf.csv')

tree = DecisionTree()


root = TreeNode()
root.column = 1
root.value = 4.5

root.left = TreeNode()
root.left.leaf = True
root.left.value = 0
root.left.name = "left"

root.right = TreeNode()
root.right.leaf = True
root.right.value = 1
root.right.name = "right"


# X = np.array(range(10))
X = np.array([1, 2])

# print(root.predict_one(X))
