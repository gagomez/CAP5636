__author__ = 'G'

def quickSort(list):

    if len(list) <= 1:
        return list

    pivot = list[0]
    list.remove(pivot)

    left = [x for x in list if x <= pivot]
    right = [x for x in list if x > pivot]

    left = quickSort(left)
    right = quickSort(right)

    left.append(pivot)
    left.extend(right)
    return left

print quickSort([4, 1, 3, 2])

