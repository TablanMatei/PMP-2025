#Exercitiul 1:
#Calculele sunt justificate pe foaie, poza atasata.

#Exercitiul 1 a)
sensibility = 0.95
specificity = 0.90
disease = 0.01

probability = (sensibility * disease)/((sensibility * disease) + ((1-specificity)* (1-disease)))
print(probability) #0.08755760368663597

#Exercitiul 1 b)
specificity = 1-(475/49500)
probability = (sensibility * disease)/((sensibility * disease) + ((1-specificity)* (1-disease)))
print(probability) #0.5

