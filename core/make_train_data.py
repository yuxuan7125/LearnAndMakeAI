import random
import pandas as pd
import os


def make_train_data(filepath: str="train_data/train_data.csv", amount: int=30) -> None:
    if amount < 10:
        raise ValueError("Amont請大於10\n")
    
    data = []

    for x in range(amount):
        for y in range(amount):

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

    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    pd.DataFrame(data).to_csv(filepath, index=None, header=None)

make_train_data()