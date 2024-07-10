import lof_hyper
import oSVM_hyper
import pandas as pd
import threading
import time
import datetime


mutex_log = threading.Lock()
best_accuracy = [0,0,0,True]


def log_metrics(metricas,q_value,n_neighbors_value,protect_anomaly_detector):
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
    log.write("Valores de q: "+ str(q_value)+ " n_neighbors: "+ str(n_neighbors_value)+ " protect_anomaly_detector: "+ str(protect_anomaly_detector)+ "\n") 

    log.write("\tAccuracy: "+ str(accuracy)+ "\n")
    log.write("\tPrecision: "+ str(precision)+ "\n")
    log.write("\tRecall: "+ str(recall)+ "\n")
    log.write("\tFalse Positive Rate: "+ str(false_positive_rate)+ "\n")
    log.write("\tF1: "+ str(f1)+ "\n")
    log.write("====================================================\n")

    if accuracy > best_accuracy[0]:
        best_accuracy[0] = accuracy
        best_accuracy[1] = q_value
        best_accuracy[2] = n_neighbors_value
        best_accuracy[3] = protect_anomaly_detector
        print("🟢 Mejor accuracy: ", accuracy ," con q: ", q_value, ", n_neighbors: ", n_neighbors_value, " y protect_anomaly_detector: ", protect_anomaly_detector)
        log.write("🟢 Mejor accuracy: "+ str(accuracy) + " con q: "+  str(q_value) + ", n_neighbors: "+ str(n_neighbors_value) + " y protect_anomaly_detector: "+ str(protect_anomaly_detector)+ "\n")
    log.close()
    mutex_log.release()

def hiperparametrizacion(dataset, q_value, n_neighbors_value, protect_anomaly_detector):
    # ct stores current time
    ct = datetime.datetime.now()
    print("🔴 ",ct," | Iniciando la prueba para los valores q: ", q_value, ", n_neighbors: ", n_neighbors_value, " y protect_anomaly_detector: ", protect_anomaly_detector)
    inicio = time.time()
    # Inicializa el modelo
    model = lof_hyper.AnomalyDetectionLOF(dataset, n_neighbors_value, q_value, protect_anomaly_detector)

    # Entrena el modelo
    metricas = model.train()
    fin = time.time()
    ct = datetime.datetime.now()
    print("🟢 ",ct," | Fin de la prueba para los valores anteriores q: ", q_value, ", n_neighbors: ", n_neighbors_value, " y protect_anomaly_detector: ", protect_anomaly_detector, " | Tiempo: ", (fin - inicio)/60, " minutos")
    log_metrics(metricas,q_value,n_neighbors_value,protect_anomaly_detector)

def main():
    print("ℹ️ Inicio de la hiperparametrizacion")

    parametros = []
    filename = 'dataset_adapted_pca_hiper.csv' # Mix 50% normal and 50% anomaly
    log = open("log.txt", "w")

    log.close()

    dataset = pd.read_csv(filename, index_col=False)

    for i in range(97, 89, -1):
        for j in range(30, 5, -1):
            parametros.append((i/100, j, False))

    threads = []
    while(len(parametros) > 0):
        for i in range(12):
            if len(parametros) == 0:
                break
            q_value, n_neighbors_value, protect_anomaly_detector = parametros.pop(0)
            # Create a thread for each test
            thread = threading.Thread(target=hiperparametrizacion, args=(dataset, q_value, n_neighbors_value, protect_anomaly_detector))
            thread.start()
            threads.append(thread)

        # Wait for the thread to finish
        for thread in threads:
            thread.join()
            threads.remove(thread)

if __name__ == "__main__":
    main()
    print("ℹ️ Fin de la hiperparametrizacion")