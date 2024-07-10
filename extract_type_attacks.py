import pandas as pd

filename = '../NF-CSE-CIC-IDS2018-v2/data/NF-CSE-CIC-IDS2018-v2.csv'
attacks = {}
chunksize = 10 ** 5
contador = 0


# Read the dataset
for chunk in pd.read_csv(filename, chunksize=chunksize):
    for index, row in chunk.iterrows():
        # Check if the row is malicious or benign
        if row['Label'] == 0: # Row benign
            continue
        if row['Attack'] in attacks:
            attacks[row['Attack']] += 1
        else:
            attacks[row['Attack']] = 1
        contador += 1
        if contador % 100000 == 0:
            print(contador)


# Print the number of each type of attack
for attack in attacks:
    print(attack, attacks[attack])
