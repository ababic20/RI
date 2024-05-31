import math
import matplotlib.pyplot as plt

#aktivacijska funkcija koja se koristi za normalizaciju izlaza neurona
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
#računanje pogreške između stvarne i predviđene
def mean_absolute_error(y_true, y_pred):
    return sum(abs(true_val - pred_val) for true_val, pred_val in zip(y_true, y_pred)) / len(y_true)

tezine_ulaz_skriveni = [
    [0.2, 0.8],  
    [0.6, 0.4],  
    [0.3, 0.7],  
    [0.1, 0.9],
    [0.5, 0.5],
    [0.4, 0.6],       
]

tezine_skriveni_izlaz = [
    [0.7, 0.2, 0.5, 0.8, 0.4],
    [0.3, 0.6, 0.9, 0.1, 0.7]
] 

ulazi = [1.0, 0.5]

stvarne_vrijednosti = [0.21, 0.47] 

#izračunavanje izlaza za sloj u mreži
def calculate_layer_output(weights, inputs):
    return [sigmoid(sum(w * i for w, i in zip(weight_set, inputs))) for weight_set in weights]

#prosljeđivanje kroz mrežu
def forward(ulazi):
    skrivene_vrijednosti = calculate_layer_output(tezine_ulaz_skriveni, ulazi)
    izlazne_vrijednosti = calculate_layer_output(tezine_skriveni_izlaz, skrivene_vrijednosti)
    return izlazne_vrijednosti
#izračun izlaza mreže i pogreške
izlazi = forward(ulazi)
print("Izlazi mreže:", izlazi)
pogreska = mean_absolute_error(stvarne_vrijednosti, izlazi)
print("Pogreška:", pogreska)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(stvarne_vrijednosti, label='Stvarne vrijednosti', marker='o')
plt.plot(izlazi, label='Predviđene vrijednosti', marker='x')
plt.title('Stvarne vs predviđene')
plt.xlabel('Izlazni neuron')
plt.ylabel('Vrijednost')
plt.legend()

plt.subplot(1, 2, 2)
errors = [abs(true - pred) for true, pred in zip(stvarne_vrijednosti, izlazi)]
plt.bar(range(len(errors)), errors, color='red')

for i, error in enumerate(errors):
    plt.text(i, error, f'{error:.2f}', ha='center', va='bottom')

plt.xticks(range(len(errors)), [f'Neuron {i+1}' for i in range(len(errors))])

plt.title('Predviđena pogreška')
plt.xlabel('Izlazni neuron')
plt.ylabel('Pogreška')

plt.tight_layout()
plt.show()
