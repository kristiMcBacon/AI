#IMPORT LIBRERIE--------------------------------------
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time
from sklearn.metrics import f1_score
import numpy as np



"""
1.  Scrivere la funzione Python decisionTreeLearner che prende come input:
        - il training set (X,y), 
        - il criterio c (gini o entropia), 
        - ccp_alpha per costruire un albero di decisione T da (X,y) 
    con i valori specificati di criterion e ccp_alpha, e restituisce T (fare riferimento a 
    help(sklearn.tree._tree.Tree) per gli attributi dell'oggetto Albero).
"""
# Decision tree learner
def decisionTreeLearner(X, y, c='gini', ccp_alpha=0.0):
    """
    Costruisce un albero decisionale utilizzando un criterio specifico e ccp_alpha.

    :param X: DataFrame o array con le feature (variabili indipendenti).
    :param y: Series o array con la variabile target.
    :param c: Criterio di divisione (default: 'gini', opzioni: 'gini' o 'entropy').
    :param ccp_alpha: Valore di pruning (default: 0.0, nessun pruning),  controlla la potatura basata sulla complessità, riduce la dimensione dell'albero per evitare il sovra-allenamento
    :return: Albero decisionale appreso (DecisionTreeClassifier).
    """
    # Costruisce l'albero decisionale con i parametri specificati:
    #   - criterion: Specifica il criterio di split ('gini' per l'impurità di Gini, 'entropy' per il guadagno di informazione).
    #   - ccp_alpha: Regola la potatura basata sulla complessità (valori maggiori aumentano la potatura).
    T = DecisionTreeClassifier(criterion=c, ccp_alpha=ccp_alpha, random_state=42)

    # Addestra l'albero sui dati forniti. X rappresenta le feature, mentre y rappresenta il target.
    T.fit(X, y)

    return T

"""
2.  Scrivere la funzione Python showTree che prende come input l'albero di decisione T e 
    traccia l'albero (usa sklearn.tree.plot_tree) e stampa le informazioni (numero di 
    nodi e numero di foglie) del T appreso
"""
def showTree(T):
    """
    Traccia l'albero decisionale e stampa informazioni sull'albero appreso.

    :param T: Albero decisionale appreso (istanza di DecisionTreeClassifier).
    """
    # Traccia l'albero decisionale
    """
    Rappresenta graficamente l'albero di decisione, parametri principali:
        - T: 
        - filled: 
        - rounded: 
        - class_names: 
        - feature_names: 
    """

    plt.figure(figsize=(20, 15))  # Dimensione del grafico
    plot_tree(
        T,                                  # Albero appreso
        filled=True,                        # Colora i nodi in base alla classe predominante
        rounded=True,                       # Angoli arrotondati per una migliore leggibilità
        class_names=True,                   # Etichette delle classi nel target
        feature_names=T.feature_names_in_,  # Nomi delle feature usate nei nodi di split
        fontsize=5,                         # dimensione del testo
        # max_depth=4,                        # Mostra solo i primi 3 livelli
        proportion=True,                    # Mostra le proporzioni delle classi rispetto al totale delle istanze nel nodo
        impurity=False,                     # Non mostra entropy o gini
    )
    plt.title("Decision Tree")
    plt.show()

    # Stampa informazioni sull'albero
    print("\n=======================================================================")
    print("Informazioni sull'albero:")
    print(f" - Numero totale di nodi: {T.tree_.node_count}")
    print(f" - Numero di foglie: {sum(T.tree_.children_left == -1)}")
    # print(f" - Profondità massima dell'albero: {T.tree_.max_depth}")

    # Usa help per visualizzare gli attributi della classe Tree
    # print("\nInformazioni su sklearn.tree._tree.Tree:")
    # Mostra la documentazione della classe Tree
    # help(sklearn.tree._tree.Tree)  

    #print(f"\nNumero totale di nodi: {num_nodes}")
    #print(f"Numero di foglie: {num_leaves}")

    print("\n=======================================================================")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# è in piu, per trovare migliore ccp_alpha
def find_best_ccp_alpha(X, y, cv=5, random_state=42):
    """
    Trova il valore ottimale di ccp_alpha utilizzando la convalida incrociata.

    :param X: DataFrame o array con le variabili indipendenti.
    :param y: Series o array con la variabile target.
    :param cv: Numero di fold per la convalida incrociata (default: 5).
    :param random_state: Seed per garantire la riproducibilità (default: 42).
    :return: Valore ottimale di ccp_alpha.
    """
    # Calcola i valori di ccp_alpha con il pruning path
    clf = DecisionTreeClassifier(random_state=random_state)
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas

    best_score = 0
    best_alpha = 0

    # Prova ogni valore di ccp_alpha
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=alpha)
        scores = cross_val_score(clf, X, y, cv=cv).mean()
        if scores > best_score:
            best_score = scores
            best_alpha = alpha

    return best_alpha
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
    Minimal Cost-complexity Pruning:
        - Utilizza gli stessi dati per far crescere l'albero e per potare l'albero
        - Sia R(T) l'errore di classificazione dell'albero T, alfa sia un parametro di input, foglie(T) sia il numero di foglie in T.
        - Definiamo il costo Ralpha(T)= R(T)+alfa*foglie(T).
        - La potatura di complessità minima del costo trova il sottoalbero di T che minimizza Ralpha(T).
    
        - La complessità dei costi di un singolo nodo t è Ralpha(t)=R(t)+alfa
        - Tt è un sotto-albero di T radicato nel nodo t
        - In generale R(Tt)<=R(t). Definiamo l'alfa efficace (alphaeff) come il valore di alfa quando Ralpha(Tt)=Ralpha(t) che è:
                        R(Tt)+alpha_eff*foglie(Tt)= R(t)+alpha_eff
        Pertanto, alphaeff=(R(t)-R(Tt))/(leaves(Tt)-1)
        - Un nodo non terminale con il valore più piccolo di alphaeff è l'anello più debole e sarà rimosso. 
        Questo processo si interrompe quando l'alphaeff minimo del l'albero potato è maggiore del parametro ccp_alpha.
"""
"""
    3.  Scrivi la funzione Python decisionTreeF1 che prende come input un set di test (XTest, YTest) e un albero 
        decisionale T e restituisce il punteggio weigthedf1 calcolato sulle previsioni prodotte da T su XTest.
"""
def decisionTreeF1(XTest, YTest, T):
    """
    Calcola il punteggio F1 ponderato sulle previsioni del modello.

    :param XTest: Dati di test (features).
    :param YTest: Dati di test (target).
    :param T: Albero decisionale appreso.
    :return: Punteggio F1 ponderato.
    """
    # Effettua le previsioni
    YPred = T.predict(XTest)
    
    # Calcola il punteggio F1 ponderato
    weightedf1 = f1_score(YTest, YPred, average='weighted')    #weighted tiene conto dell'importanza delle diverse classi nel set di dati
    
    return weightedf1

"""
    4.  Scrivi la funzione Python "determineDecisionTreekFoldConfiguration" che prende come input
        la validazione incrociata a 5 punti per determinare la migliore configurazione rispetto al 
        criterio (gini o entropia) e ccp_alpha (che si trova tra 0 e 0,05 con il passo 0,001). 
        La migliore configurazione è determinata rispetto al peso del F1. 
        La funzione restituisce il criterio, ccp_alpha e best weigthed della migliore configurazione.
"""
def determineDecisionTreekFoldConfiguration(X, y):
    """
    Determina la configurazione migliore di criterio e ccp_alpha tramite 5-fold CV.
    
    :param X: Dati di training (features).
    :param y: Dati di training (target).
    :return: Il criterio migliore, il miglior ccp_alpha e il punteggio F1 ponderato della configurazione migliore.
    """
    print("\nInizio determineDecisionTreekFoldConfiguration...")
    # Registra il tempo di inizio
    start_time = time.time()

    # Criteri da esplorare
    criteria = ['gini', 'entropy']
    # Intervallo di valori per ccp_alpha
    ccp_alpha_values = np.arange(0, 0.05, 0.001)
    
    best_f1 = 0
    best_criterion = None
    best_ccp_alpha = None
    
    # Inizializza Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Itera su criteri e valori di ccp_alpha
    for c in criteria:
        for alpha in ccp_alpha_values:
            f1_scores = []
            
            # Cross-validation a 5 fold
            for train_idx, test_idx in skf.split(X, y):
                XTrain, XTest = X.iloc[train_idx], X.iloc[test_idx]
                yTrain, yTest = y.iloc[train_idx], y.iloc[test_idx]
                
                # Addestra un albero decisionale
                T = decisionTreeLearner(XTrain, yTrain, c=c, ccp_alpha=alpha)
                
                # Calcola il punteggio F1 ponderato
                f1 = decisionTreeF1(XTest, yTest, T)
                f1_scores.append(f1)
            
            # Calcola la media dei punteggi F1
            mean_f1 = np.mean(f1_scores)
            
            # Aggiorna la configurazione migliore se necessario
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_criterion = c
                best_ccp_alpha = alpha
    
    # Registra il tempo di fine
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nFine determineDecisionTreekFoldConfiguration. Tempo impiegato: {elapsed_time:.4f} secondi.")
    return best_criterion, best_ccp_alpha, best_f1