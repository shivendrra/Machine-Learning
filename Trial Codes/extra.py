pic_array = ['aabba', 'aabba', 'aaacb']

for i in range(len(pic_array)):
  for j in range(len(pic_array[i])):
    elem = pic_array[i][j]
    if elem == 'a':
      print('green')
    if elem == 'b':
      print('blue')
    if elem == 'c':
      print('red')

# ----------------------------

# num = str(649578)
# sum = 0
# # in_num = int(input('enter a no '))
# for i in range(len(num)):
#   print(num[i])
#   if num[i] == '0' or num[i] == '4' or num[i] == '6' or num[i] == '9':
#     print('added 1')
#     sum = sum + 1
#   elif num[i] == '8' or num[i] == '2':
#     print('added 2')
#     sum = sum + 2
#   else:
#     print('igonred')
#     sum = sum + 0

# print('sum:', sum)