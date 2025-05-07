import numpy as np





array1 = [
    3, 4, 5
]

array2 = [
    [1,2,3],
    [4,5,6]
]


array1 = np.array(array1)
array2 = np.array(array2)




print(array1)
print(array2)

print("sum, array2", np.sum(array2))
print(np.sum(array2, keepdims=True))
print(np.sum(array2, axis=1, keepdims=True))