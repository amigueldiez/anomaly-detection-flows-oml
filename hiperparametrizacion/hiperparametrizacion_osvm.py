import lof_hyper
import oSVM_hyper
import pandas as pd
import threading
import time
import datetime


mutex_log = threading.Lock()
best_accuracy = [0,0,0]


def log_metrics(metricas,nu_value, umbral_score):
    tp = metricas[0]
    tn = metricas[1]
    fp = metricas[2]
    fn = metricas[3]

    print("🟢 Resultados de la prueba")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("Accuracy: ", accuracy)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    print("Precision: ", precision)
    recall = tp / (tp + fn)
    print("Recall: ", recall)
    false_positive_rate = fp / (fp + tn)
    print("False Positive Rate: ", false_positive_rate)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    print("F1: ", f1)

    # Save values in a log file
    mutex_log.acquire()
    log = open("log.txt", "a")

    log.write("====================================================\n")
    log.write("Valores de nu_value: "+ str(nu_value)+ " umbral_score: "+ str(umbral_score)+  "\n") 

    log.write("\tAccuracy: "+ str(accuracy)+ "\n")
    log.write("\tPrecision: "+ str(precision)+ "\n")
    log.write("\tRecall: "+ str(recall)+ "\n")
    log.write("\tFalse Positive Rate: "+ str(false_positive_rate)+ "\n")
    log.write("\tF1: "+ str(f1)+ "\n")
    log.write("====================================================\n")
    if accuracy > best_accuracy[0]:
        best_accuracy[0] = accuracy
        best_accuracy[1] = nu_value
        best_accuracy[2] = umbral_score
        print("🟢 Mejor accuracy: ", accuracy, " con nu_value: ", nu_value, " y umbral_score: ", umbral_score)
        log.write("🟢 Mejor accuracy: "+ str(accuracy)+ " con nu_value: "+ str(nu_value)+ " y umbral_score: "+ str(umbral_score)+ "\n")
    
    log.close()
    mutex_log.release()



def hiperparametrizacion(dataset, nu_value, umbral_score):
    # ct stores current time
    ct = datetime.datetime.now()
    print("🔴 ",ct," | Iniciando la prueba para los valores nu: ", nu_value, " y umbral_score: ", umbral_score)
    inicio = time.time()
    # Inicializa el modelo
    model = oSVM_hyper.AnomalyDetectionoSVM(dataset,nu_value, umbral_score)

    # Entrena el modelo
    metricas = model.train()
    fin = time.time()
    ct = datetime.datetime.now()
    print("🟢 ",ct," | Fin de la prueba para los valores anteriores nu: ", nu_value, " y umbral_score: ", umbral_score, " | Tiempo: ", (fin - inicio)/60, " minutos")
    log_metrics(metricas, nu_value, umbral_score)

def main():
    print("ℹ️ Inicio de la hiperparametrizacion")

    parametros = []
    filename = 'dataset_adapted_pca_dos_ataq_5d.csv' # Mix 50% normal and 50% anomaly
    log = open("log.txt", "w")

    log.close()

    dataset = pd.read_csv(filename, index_col=False)

    for i in range(0, 99):
        for j in range(0,1000):
            parametros.append((i/100, j/1000))
            parametros.append((i/100, -j/1000))


    threads = []
    while(len(parametros) > 0):
        for i in range(12): # Number of threads
            if len(parametros) == 0:
                break
            n_value, umbral_score= parametros.pop(0)
            # Create a thread for each test
            thread = threading.Thread(target=hiperparametrizacion, args=(dataset, n_value, umbral_score))
            thread.start()
            threads.append(thread)

        # Wait for the thread to finish
        for thread in threads:
            thread.join()
            threads.remove(thread)

    print("🟢 Mejor accuracy: ", best_accuracy[0], " con nu_value: ", best_accuracy[1], " y umbral_score: ", best_accuracy[2])

if __name__ == "__main__":
    main()
    print("ℹ️ Fin de la hiperparametrizacion")