import random
import pandas as pd


def make_train():
    data = []

    for x in range(30):
        for y in range(30):

            rd = random.randint(1,3)
            if rd!=1 :
                continue

            ans = x**2 + 3*x - 2*y**2 + y - 3
            rd = random.randint(1,2)
            if rd==1 :
                ans+=1
            else :
                ans-=1

            data.append([x,y,ans])

    return data