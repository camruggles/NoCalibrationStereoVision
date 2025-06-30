import matplotlib.pyplot as plt

a = (0,0)
b = (0.015, 0.0)
c = (0.015, 0.015)
d = (0.0, 0.015)
arr = [a,b,c,d]
id_dict =  {}
id_dict[0] = arr
count = 0
for y in range(8):
    for x in range(11):
        if count % 2 == 0:
            id = count //2
            [a,b,c,d] = id_dict[0]
            a1,a2 = a
            b1,b2 = b 
            c1, c2 = c
            d1, d2 = d
            id_dict[id] = [(a1 + x * 0.02, a2 + y * 0.02), (b1 + x * 0.02, b2 + y * 0.02), (c1 + x * 0.02, c2 + y * 0.02), (d1 + x * 0.02, d2 + y * 0.02)]
            print(id, id_dict[id])
        count += 1
    # import pdb; pdb.set_trace()

# print(id_dict)
# x_coords = []
# y_coords = []
# # Iterate over the dictionary items.
# for key, tuples in id_dict.items():
#     for tup in tuples:
#         # Assuming the first element is x and the second is y.
#         x, y = tup[0], tup[1]
#         x_coords.append(x)
#         y_coords.append(-y)


# x_coords.append(0.28)
# y_coords.append(-0.22)
# # Plot the coordinates using a scatter plot.
# plt.scatter(x_coords, y_coords)
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Plot of all x,y coordinates')
# plt.grid(True)
# plt.show()