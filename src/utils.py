def k_retrieval(list, k, reverse=False):
    """
    Args:
        - list: list of integers
        - k: how many elements retrieve
    Returns
        The K lowest values from that list
    """

    return list.sort(reverse=reverse)[0:k + 1]
