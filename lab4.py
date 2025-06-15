from lab.lab1 import *

#IMPORT LIBRERIE--------------------------------------
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay





"""
    Confusion Matrix and Classification Report:
        Caricare il test set testDdosLabelNumeric.csv e generare le previsioni per i campioni di prova 
        utilizzando gli alberi decisionali appresi dall'intero set di formazione con il meglio. 
        Determinare e mostrare la matrice di confusione, nonché stampare il rapporto di classificazione 
        calcolato sulla previsione prodotta sui campioni di prova.
"""
# Funzione per caricare il dataset di testing e calcolare i risultati
def evaluateDecisionTree(model, test_data):
    """
    Genera le predizioni su un dataset di test pre-elaborato e calcola la matrice di confusione
    e il report di classificazione.

    :param model: Albero decisionale appreso.
    :param test_data: DataFrame del dataset di test pre-elaborato.
    """

    print("\nGenerazione delle predizioni...")
    
    # Separazione feature e target dal dataset di test
    X_test = test_data.iloc[:, :-1]  # Tutte le colonne tranne l'ultima (feature)
    y_test = test_data.iloc[:, -1]   # Ultima colonna (target)

    # Generazione delle predizioni utilizzando il modello
    y_pred = model.predict(X_test)

    # Creazione della matrice di confusione
    print("\nCreazione della matrice di confusione...")
    cm = confusion_matrix(y_test, y_pred)

    # Visualizzazione della matrice di confusione come grafico
    print("\nVisualizzazione della matrice di confusione...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Matrice di Confusione", fontsize=16)
    ax.set_xlabel("Predizioni", fontsize=14)
    ax.set_ylabel("Valori Reali", fontsize=14)
    plt.show()

    print("\nReport di classificazione:")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)



#-----------------------------------------------------
"""
    Random Forest learner + Stratified CV
    
    1) Scrivere la funzione Python determineRFkFoldConfiguration che prende come input:
        - la validazione incrociata a 5-punti: per determinare la migliore configurazione rispetto al criterio 
        (gini o entropia) di randomizzazione (sqrt o log2),  
        - dimensione bootstrap: (con max_samples variabili tra 0,7, 0,8 e 0,9), 
        - numero di alberi (variabile tra 10, 20, 30 , 40 ad 50),
        - ccp_alpha (variabile tra 0 e 0,05 con il passo 0,001). . 
    La migliore configurazione è determinata rispetto al peso medio di F1 sulle pieghe di prova. 
    La funzione restituisce:
        - criterio, randomizzazione, bootstrap, numero di alberi, ccp_alpha e media del peso dF1 della migliore configurazione 
    
    2) Impara una foresta casuale usando l'intero set di allenamento e considerando la migliore configurazione 
    identificata con determineRFkFoldConfiguration 
    
    3) Verificare l'accuratezza del modello di foresta casuale su testDdosLabelNumeric.
"""
#-----------------------------------------------------
#1) Determine the Best Random Forest Configuration
"""
    Per implementare la funzione determineRFkFoldConfiguration, che esegue una cross-validation per determinare 
    la configurazione migliore per un modello Random Forest, dobbiamo eseguire un ciclo su diverse configurazioni 
    degli iperparametri del modello e valutarlo utilizzando la metrica F1 ponderata. 
"""
def determineRFkFoldConfiguration(X, y, folds=5, seed=40):
    # Griglia degli iperparametri
    criteria = ['gini', 'entropy']
    randomizations = ['sqrt', 'log2']
    bootstrap_sizes = [0.7, 0.8, 0.9]
    num_trees = [10, 20, 30, 40, 50]
    ccp_alphas = np.arange(0, 0.051, 0.001)  # Da 0 a 0.05 con passo 0.001
    
    # Inizializza StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    # Variabili per tracciare la migliore configurazione e il punteggio
    best_weighted_f1 = -1
    best_config = None

    # Numero totale di combinazioni da esplorare
    total_combinations = len(criteria) * len(randomizations) * len(bootstrap_sizes) * len(num_trees) * len(ccp_alphas)
    current_combination = 0
    
    # Cicla su tutte le combinazioni di iperparametri
    for criterion in criteria:
        for randomization in randomizations:
            for bootstrap_size in bootstrap_sizes:
                for n_trees in num_trees:
                    for ccp_alpha in ccp_alphas:

                        current_combination += 1  # Incrementa il contatore delle combinazioni
                        
                        # Messaggio di progresso: quante combinazioni sono rimaste
                        print(f"Combinazione {current_combination}/{total_combinations}")
                        
                        # Inizializza il RandomForestClassifier con la combinazione attuale di iperparametri
                        rf = RandomForestClassifier(
                            criterion=criterion,
                            max_features=randomization,
                            bootstrap=True,
                            max_samples=bootstrap_size,
                            n_estimators=n_trees,
                            ccp_alpha=ccp_alpha,
                            random_state=seed
                        )
                        
                        # Lista per memorizzare i punteggi F1 per ogni fold
                        fold_f1_scores = []
                        
                        # Esegui la cross-validation
                        for train_index, test_index in skf.split(X, y):
                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                            
                            # Allena il modello
                            rf.fit(X_train, y_train)
                            
                            # Fai le previsioni e calcola il punteggio F1 ponderato
                            y_pred = rf.predict(X_test)
                            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                            fold_f1_scores.append(weighted_f1)
                        
                        # Calcola la media del punteggio F1 ponderato per la configurazione attuale
                        avg_weighted_f1 = np.mean(fold_f1_scores)
                        
                        # Se questa è la configurazione migliore finora, aggiorna la miglior configurazione e il punteggio
                        if avg_weighted_f1 > best_weighted_f1:
                            best_weighted_f1 = avg_weighted_f1
                            best_config = (criterion, randomization, bootstrap_size, n_trees, ccp_alpha)
    
    # Messaggio finale con la configurazione migliore
    print("\nCompletata la ricerca delle configurazioni")
    
    # Restituisci la migliore configurazione e il relativo punteggio F1 ponderato medio
    return best_config + (best_weighted_f1,)

#-----------------------------------------------------
#3) Test the Random Forest Model
def testRandomForest(rf_model, test_data):
    """
    Funzione per testare l'accuratezza del modello Random Forest sui dati di test.

    Parametri:
    - rf_model: Il modello RandomForestClassifier addestrato.
    - testpath: Il percorso del file CSV contenente i dati di test.

    Ritorna:
    - accuracy: L'accuratezza del modello sui dati di test.
    """
    print("Testing Random Forest on the testing dataset...")
    
    # Separate features and target
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Display confusion matrix (non specificato)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    disp.plot(cmap="Blues")
    plt.show()

    # Display classification report (non specificato)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Restituisci l'accuratezza
    return accuracy


#-----------------------------------------------------
