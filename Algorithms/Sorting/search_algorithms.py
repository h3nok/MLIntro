def linear_search(data, item):
    index = 0
    found = False

    while index < len(data): 
        if data[index] == item:
            found = True
        else:
            index += 1
    
    return found, index

def binary_search(data, item):
    first = 0
    last = len(data) - 1
    found = False

    while first <= last and not found:
        midpoint = (first + last) // 2

        if data[midpoint] == item:
            found = True
        else:
            if item < data[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
                
    return found

def interpolation_serach(list,x ):
    idx0 = 0
    idxn = (len(list) - 1)
    found = False
    while idx0 <= idxn and x >= list[idx0] and x <= list[idxn]:
        # Find the mid point
        mid = idx0 +int(((float(idxn - idx0)/( list[idxn] - list[idx0])) *( x - list[idx0])))
        # Compare the value at mid point with search value
        if list[mid] == x:
            found = True
            return found
        if list[mid] < x:
            idx0 = mid + 1

    return found

data = [12,13, 11, 99, 22, 55, 90]
print(binary_search(sorted(data), 99))