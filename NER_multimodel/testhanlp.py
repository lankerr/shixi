from myhanlp import myhanlp

while (True):
    str = input()
    if str == "exit":
        break
    else:
        mystr = myhanlp(str)
        print(mystr)
