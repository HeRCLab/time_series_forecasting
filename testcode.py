

# mylist = [0,1,2,1,1,1,2,1,1,1,2,1,1,1,2,15,16,0,1,2,1,1,1,2,1,1,1,2,1,1,1,2,15,16,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,2,3,4,5,6,7,8,9,10,11,12,13,100,1,2,3]
mylist = [0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 15, 16, 0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2,
          15, 16, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 99,
          21, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100, 9, 2, 3]

bptt = 16
print('len(mylist) = ', len(mylist))
# for batch, i in enumerate(range(0, len(mylist), bptt)):
#     print('i = ',i)
#     print('batch = ',batch)
#     tmplist = mylist[i:i+bptt]
#     print('tmplist = ', tmplist)
src = []
trg = []
trgout = []
# without sliding window
list_len = len(mylist)
for indx, stepindx in enumerate(range(0, len(mylist), bptt)):
    print('stepindx = ', stepindx)
    print('indx = ',indx)
    # tmplist = mylist[i:i+bptt]
    # print('tmplist = ', tmplist)


    if stepindx+bptt >= list_len:
        break

    srcsgmnt = mylist[stepindx:stepindx+bptt]
    src.append(srcsgmnt)

    trgsgmnt = mylist[stepindx:stepindx+bptt]
    trg.append(trgsgmnt)

    trgout.append(mylist[stepindx+bptt])

print('src = ', src)
print('trg = ', trg)
print('trgout = ', trgout)

