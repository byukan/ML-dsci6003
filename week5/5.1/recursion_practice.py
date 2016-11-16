class TreeNode(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def equals(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
    if root1.value != root2.value:
        return False
    return equals(root1.left, root2.left) and equals(root1.right, root2.right)


def sum_tree(root):
    """
    sum_tree(root) = sum_tree(root.left) + sum_tree(root.right) + root.value
    :param root:
    :return:
    """
    if not root:
        return 0
    return sum_tree(root.left) + sum_tree(root.right) + root.value


def print_all(root):
    """
    prints all the values in a binary tree
    :param root:
    :return:
    """
    if root:
        # preorder
        print(root.value, end=","), print_all(root.left), print_all(root.right)
        # postorder
        # print()
        # inorder


def test_tree():
    # build a tree
    #     1
    #    / \
    #   2   3
    #  /
    # 4
    t1 = TreeNode(1)
    t1.left = TreeNode(2)
    t1.right = TreeNode(3)
    t1.left.left = TreeNode(4)

    # build a tree
    #     1
    #    / \
    #   2   3
    #  /   /
    # 4   5
    t2 = TreeNode(1)
    t2.left = TreeNode(2)
    t2.right = TreeNode(3)
    t2.left.left = TreeNode(4)
    t2.right.left = TreeNode(5)

    print(sum_tree(t1), sum_tree(t2))
    print(print_all(t1))


# test_tree()


def sum_digits(n):
    """
    Write a recursive function which takes an integer and computes and sum of the digits:
    sum_digits(4502)    # returns 11
    :param n:
    :return:
    """
    if n == 0:
        return 0
    else:
        # ones digit + sum_digits on the rest
        non_ones_digits = int(n / 10)
        return n % 10 + sum_digits(non_ones_digits)


def build_coinflip_tree(k, val = ""):
    """
    takes an integer k and builds the tree containing all the possible results of flipping a coin k times The value
    at each node should be a string of the flips to get there. For example, if k is 3, your tree should look like
    something similar to this:

                        ''
                      /    \
                    /        \
                  /            \
                /                \
              H                    T
            /   \                /   \
          /       \            /       \
        HH         HT        TH         TT
       /  \       /  \      /  \       /  \
     HHH  HHT   HTH  HTT  THH  THT   TTH  TTT

    :param k:
    :return:
    """
    root = TreeNode(val)
    if k != 0:
        root.left = build_coinflip_tree(k - 1, root.value + 'H')
        root.right = build_coinflip_tree(k - 1, root.value + 'T')

    return root


def main():
    # print(sum_digits(4502))
    my_tree = build_coinflip_tree(3, "")
    print_all(my_tree)
    root = build_coinflip_tree(3, "")
    assert root.value == ""
    assert root.left.value == "H"
    assert root.left.left.value == "HH"


main()
