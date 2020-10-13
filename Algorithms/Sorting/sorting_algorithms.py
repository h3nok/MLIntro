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
data = [19,12,3,1,3,10,-2]

sorted_data = bubble_sort(data)


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
