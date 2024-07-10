# Create own dataset from the given dataset

import pandas as pd

# Parameters
filename = '../NF-CSE-CIC-IDS2018-v2/data/NF-CSE-CIC-IDS2018-v2.csv'
number_of_rows = 10000
chunksize = 10 ** 5
porcentage_of_malicious_samples = 0.25
new_filename = 'dataset_adapted.csv'
start_chuck_number = 0
types_of_attacks = [
    'SSH-Bruteforce',
    'DDoS attacks-LOIC-HTTP',
    'DDOS attack-HOIC',
    'DoS attacks-Slowloris',
    'DoS attacks-Hulk',
    'FTP-BruteForce',
    'Infilteration',
    'Bot',
    'DoS attacks-GoldenEye',
    'Brute Force -Web',
    'DoS attacks-SlowHTTPTest',
    'SQL Injection',
    'DDOS attack-LOIC-UDP',
    'Brute Force -XSS'
]



# Start of script

def get_malicious_samples_remaining():
    total = 0
    for attack in malicious_attacks:
        total += malicious_attacks[attack]
    return total

malicious_samples = round(number_of_rows * porcentage_of_malicious_samples)
normal_samples = number_of_rows - malicious_samples

# Create a dictionay from types of attacks

malicious_attacks = {}
for attack in types_of_attacks:
    malicious_attacks[attack] = malicious_samples/len(types_of_attacks)

print("📚 Quantity of malicious samples per attack: ", malicious_samples/len(types_of_attacks))

# Create a file that contains the new dataset
new_file = open(new_filename, 'w')

stop_malicious = False
stop_normal = False
write_header = True
index = 1
number_chunk = 1

# Read the dataset
for chunk in pd.read_csv(filename, chunksize=chunksize):
    if number_chunk >= start_chuck_number:
        print('⏳ Chunk number: ', number_chunk)

        # chunk is a DataFrame. To "process" the rows in the chunk:
        # Write header of dataset in csv
        if write_header:
            new_file.write(','.join(chunk.columns.tolist()) + '\n')
            write_header = False
        for index, row in chunk.iterrows():
            # Check if the row is malicious or benign
            if row['Label'] == 0 and normal_samples > 0: # Row benign
                new_file.write(row.to_csv(index=False, lineterminator=',', header=False)+ '\n')
                normal_samples -= 1

            elif row['Attack'] in types_of_attacks and malicious_attacks[row['Attack']] > 0: # Row malicious
                new_file.write(row.to_csv(index=False, lineterminator=',',header=False)+ '\n')
                malicious_attacks[row['Attack']] -= 1
                if get_malicious_samples_remaining() % 100 == 0:
                    print('Malicious samples remaining: ', get_malicious_samples_remaining())
                    print(malicious_attacks)
                

        if get_malicious_samples_remaining() == 0 and normal_samples == 0:
            break

    number_chunk += 1
    
new_file.close()

# Read dataset adapted
dataset_new = pd.read_csv(new_filename, index_col=False)
# Shuffle the dataset
dataset_new = dataset_new.sample(frac=1).reset_index(drop=True)
# Save the dataset
dataset_new.to_csv(new_filename, index=False)


print('✅ Dataset created')
print("🔎 Checking the quality...")

# Check quality of new dataset
# Warning: only adapated for small datasets
dataset_pd = pd.read_csv(new_filename,index_col=False)

print(dataset_pd.head()) 
# Count the number of malicious and normal samples
malicious = 0
normal = 0
type_attack = {}
for index, row in dataset_pd.iterrows():
    if row['Label'] == 0:
        normal += 1
    else:
        malicious += 1

    if row['Attack'] not in type_attack:
        type_attack[row['Attack']] = 1
    else:
        type_attack[row['Attack']] += 1

print('Number of malicious samples: ', malicious)
print('Number of normal samples: ', normal)

print('Type of attacks: ')
print(type_attack)


            