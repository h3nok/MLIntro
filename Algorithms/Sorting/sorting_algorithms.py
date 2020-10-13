def bubble_sort(data):
    """ Bubble sort is the simplest and slowest algorithm used for sorting. It is designed in a way
        that the highest value in its list bubbles its way to the top as the algorithm loops through
        iterations. As its worst-case performance is O(N2) and is suited only for small datasets 

    Args:
        data ([lis]): a list of items to sort

    Returns:
        list: sorted list
    """
    assert isinstance(data, list)

    last_element_index = len(data) - 1
    for pass_number in range(last_element_index,0,-1):
        for i in range(pass_number):
            if data[i] > data[i+1]:
                data[i], data[i+1] = data[i+1], data[i]
    return data


def insertion_sort(data):
    """The basic idea of insertion sort is that in each iteration, we remove a data point from the
       data structure we have and then insert it into its right position. That is why we call this the
       insertion sort algorithm. In the first iteration, we select the two data points and sort them.
       Then, we expand our selection and select the third data point and find its correct position,
       based on its value. The algorithm progresses until all the data points are moved to their
       correct positions.

    Args:
        data (list): a list of items to sort 

    returns:
        list: sorted list 
    """

    for i in range(1, len(list)):
        j = i-1
        next_element = data[i]
        while(list[j] > next_element) and (j >= 0):
            data[j+1] = data[j]
            j = j - 1
        data[j+1] = next_element
    
    return data


def merge_sort(data):
    """1. It divides the input list into two equal parts
       2. It uses recursion to split until the length of each list is 1
       3. Then, it merges the sorted parts into a sorted list and returns it

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    if len(data) > 1:
        mid = len(data) // 2
        left = data[:mid]
        right = data[mid:]

        merge_sort(left)
        merge_sort(right)

        a = 0
        b = 0
        c = 0

        while a < len(left) and b < len(right):
            if left[a] < right[b]:
                data[c] = left[a]
                a += 1
            else:
                data[c] = right[b]
                b += 1
            c += 1

        while a < len(left):
            data[c] = left[a]
            a += 1
            c += 1
        while b < len(right):
            data[c] = right[b]
            b += 1
            c += 1

    return data 

def shell_sort(data):
    distance = len(data) // 2
    while distance > 0:
        for i in range(distance, len(data)):
            temp = data[i]
            j = i
            # Sort the sub list for this distance
            while j >= distance and data[j - distance] > temp:
                data[j] = data[j - distance]
                j = j-distance
            data[j] = temp

        # Reduce the distance for the next element
        distance = distance//2

    return data 

def selection_sort(data):
    for fill_slot in range(len(data)-1, 0, -1):
        max_index = 0 

        for location in range(1, fill_slot + 1):
            if data[location] > data[max_index]:
                max_index = location
        data[fill_slot], data[max_index] = data[max_index], data[fill_slot]
    return data
