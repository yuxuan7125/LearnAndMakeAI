import random


def make_train(amount: int=30) -> list[list[int, int, int]]:
    if amount < 10:
        raise ValueError("Amont請大於10")
    
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

    return data