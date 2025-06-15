#IMPORT LIBRERIE--------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score


"""
    Cos’è il Seed?
    Il seed è un parametro utilizzato per inizializzare il generatore di numeri casuali in programmazione. 
    Nel contesto del machine learning, impostare un seed serve a:
    - Garantire che le operazioni casuali (come la suddivisione dei dati in fold) producano sempre gli stessi risultati ad ogni 
    esecuzione del programma. Questo rende il processo riproducibile, una proprietà essenziale per esperimenti scientifici e per il debugging.
    - Senza un seed fisso, i risultati variano tra le esecuzioni, rendendo difficile confrontare o validare i risultati.

    Stratified K-Fold Cross-Validation:
    La Stratified K-Fold Cross-Validation è una tecnica di validazione incrociata che:
    - Garantisce che la distribuzione della variabile target (y) in ciascun fold sia proporzionale alla sua distribuzione nell’intero dataset. 
    Questo è particolarmente utile per dataset sbilanciati.
    - Divide il dataset in K fold, quindi allena e valuta il modello su ciascun fold, assicurando che ogni dato venga testato esattamente una volta.

    -----------------------------------------------------------

    Punti Chiave
    - Perché usare Stratified K-Fold?
    Garantisce che ogni fold abbia una distribuzione simile della variabile target, particolarmente importante per dataset sbilanciati.

    - Perché impostare un Seed?
    Per garantire la riproducibilità. Un seed fisso assicura che gli stessi fold vengano generati ad ogni esecuzione.

    - Scelta dei Parametri:
        - folds: In genere, si usano 5 o 10 fold in pratica.
        - shuffle=True: Garantisce che i dati vengano mescolati casualmente prima di essere suddivisi in fold.

    Questa implementazione prepara il dataset per un addestramento e una valutazione robusti, 
    mantenendo il bilanciamento della classe in ogni fold e permettendo la riproducibilità tramite il seed.
"""


#Stratified K-fold CV
def stratifiedKfold(X, y, folds, seed):
    """
    Esegui la Stratified K-Fold Cross-Validation.

    :param X: DataFrame o array contenente le variabili indipendenti (features).
    :param y: Series o array contenente la variabile target
    :param folds: Numero di fold per la validazione incrociata.
    :param seed: Seed per il generatore di numeri casuali (garantisce la riproducibilità).
    :return: Liste di training set e testing set per X e y per ciascun fold.

    Output della Funzione:
        - ListXTrain: Contiene i training set delle variabili indipendenti (features) per ciascun fold.
        - ListXTest: Contiene i testing set delle variabili indipendenti (features) per ciascun fold.
        - ListyTrain: Contiene i training set della variabile target per ciascun fold.
        - ListyTest: Contiene i testing set della variabile target per ciascun fold.
    """
    
    # Inizializza StratifiedKFold
    # n_splits=folds:       Specifica il numero di fold.
    # shuffle=True:         Mescola i dati prima di suddividerli, mantenendo però la proporzione della variabile target.
    # random_state=seed:    Imposta il seed per garantire che il mescolamento sia riproducibile.
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # Creazione liste per memorizzare le suddivisioni di train/test per ciascun fold
    ListXTrain, ListXTest, ListyTrain, ListyTest = [], [], [], []

    # Suddivide in fold
    # La funzione split divide gli indici delle righe di X e y in folds bilanciati sulla distribuzione delle classi
    for train_index, test_index in skf.split(X, y):     
        # Estrae i dati di training e testing per ciascun fold
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # Aggiungi le suddivisioni alle rispettive liste
        ListXTrain.append(X_train)
        ListXTest.append(X_test)
        ListyTrain.append(y_train)
        ListyTest.append(y_test)

    return ListXTrain, ListXTest, ListyTrain, ListyTest