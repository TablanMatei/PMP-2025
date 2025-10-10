
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import random as rd

# Exercitiul 1
# 1.a)
def a():

    nr_bile_rosii= 3
    nr_bile_albastre= 4
    nr_bile_negre= 2
    Urna = []

    for i in range (nr_bile_rosii):
        Urna.append("r")
    for i in range (nr_bile_albastre):
        Urna.append("a")
    for i in range (nr_bile_negre):
        Urna.append("n")

    dice = rd.randrange(1,6)

    if dice==2 or dice==3 or dice==5: Urna.append("n")
    elif dice==6: Urna.append("r")
    else: Urna.append("a")

    rd.shuffle(Urna)
    extragere = rd.randrange(0,len(Urna))
    if (Urna[extragere]=='r'):
        #print("S-a extras o bila rosie")
        return 'r'
    if (Urna[extragere]=='a'):
        #print("S-a extras o bila albastra")
        return 'a'
    if (Urna[extragere]=='n'):
        #print("S-a extras o bila neagra")
        return 'n'
'''
# 1.b)

    A="se extrage o bila rosie"
    B1="Zar este 2 sau 3 sau 5"
    B2="Zar este 6"
    B3="Zar este 1 sau 4"
   
    P(A)=?     Formula probabilitatii totale:
    P(A) = P(A|B1) P(B1) + P(A|B2) P(B2) + P(A|B3) P(B3)
         = 3/10 x 3/6 + 4/10 x 1/6 + 3/10 x 2/6
         = 19/60 = 0.316
  '''

for incercari in range (0,5):
    succes=0
    nr_experimente= 1000000
    for i in range(nr_experimente):
        if a() == 'r': succes = succes+1
    print(succes/nr_experimente)
    
    
  # 1.c)  Obtin cam 0.30 in loc de 0.31



#Exercitiul 2
import numpy as np
import matplotlib.pyplot as plt


#2.1
lamba_mult = [1, 2, 5, 10]

for lam in lamba_mult:
    samples = np.random.poisson(lam, 1000)
    values, counts = np.unique(samples, return_counts=True)

    plt.bar(values, counts / sum(counts), color='green', edgecolor='black')
    plt.title(f'Distributia Poisson')
    plt.xlabel('Valori')
    plt.ylabel('Probabilitate')
    plt.show()


#2.2
lambdas_random = np.random.choice(lamba_mult, size=1000, replace=True)
samples = np.random.poisson(lambdas_random)
values, counts = np.unique(samples, return_counts=True)

plt.bar(values, counts / sum(counts), color='purple', edgecolor='black')
plt.title('Distributia Poisson randomizată')
plt.xlabel('Valori')
plt.ylabel('Probabilitate')
plt.show()

#2.2a) - am atasat histogramele
#2.2b) - Observ ca la Distributia Randomizata probabilitatea este mai mare pentru valori mai mici
# si scade gradual