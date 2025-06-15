"""
    Attivare ambiente virtuale con: 
    .\envProgettoAI\Scripts\activate
"""
"""
DUBBI:
    - Perche si segono 5 bins? 
    data[target_column].hist(bins=5, ax=ax)  # Crea l'istogramma con 5 intervalli

    -Criterio con cui si eliminano le colonne:
    def removeColumns(data, cols):

"""

#IMPORT FILE, uno per ogni Lab Activity-------------------------------------------------------------
from lab.lab1 import * 
from lab.lab2 import * 
from lab.lab3 import *
from lab.lab4 import * 


#___________________________________________1 lab activity___________________________________________
# load data
print("\n=======================================================================")
trainpath="trainDdosLabelNumeric.csv"
data=load(trainpath)
# ottieni la forma (numero di righe e colonne) del DataFrame e la stampa
shape=data.shape
print("\nShape CSV:")
print(shape)
# Stampa le prime 5 righe del DataFrame per dare un'idea dei dati caricati
print("=======================================================================")
print("\nPrime 5 righe CSV:")
print(data.head())
# Stampa i nomi delle colonne nel DataFrame
print("=======================================================================")
print("\nNomi colonne CSV:")
print(data.columns)
print("=======================================================================")

#-----------------------------------------------------
# Pre-elaboration
# ottieni la lista delle colonne
cols = list(data.columns.values)
preElaborationData(data,cols)
# Drop useless columns in PANDAS
data,removedColumns=removeColumns(data,cols)
print("\nColonne rimosse:")
print(removedColumns)
print("=======================================================================")
preElaborationClass(data, 'Label')



#___________________________________________2 lab activity___________________________________________
#stratified K-fold CV

print("\n=======================================================================")
# Ottieni la lista di tutte le colonne del DataFrame 'data'
cols = list(data.columns.values)
# Crea una lista delle variabili indipendenti (tutte le colonne tranne l'ultima)
independentList = cols[0:data.shape[1] - 1]     # Utilizziamo un slicing per escludere l'ultima colonna (che di solito è la variabile target)
                                                # 'data.shape[1]' restituisce il numero totale di colonne
print("Variabili indipendenti:")
print(independentList)
# Specifica il nome della variabile target
target='Label'
# Seleziona le feature (variabili indipendenti) dal DataFrame
X=data.loc[:, independentList];                 # Usa 'loc' per selezionare le colonne specificate da 'independentList'
# Seleziona la variabile target (valore di classe) dal DataFrame
y=data[target]
# Imposta il numero di fold per la Stratified K-Fold Cross-Validation
folds=5
# Imposta il seed per garantire la riproducibilità del processo di shuffling
seed=40

# Esegui Stratified K-Fold CV
ListXTrain, ListXTest, ListyTrain, ListyTest = stratifiedKfold(X, y, folds, seed)
# Il comando print(data.head()) stampa le prime 5 righe del DataFrame data.
# La funzione head() di Pandas è usata per visualizzare un'anteprima del DataFrame.
print("\n=======================================================================")
print("Prime 5 righe del DataFrame:")
print(data.head())



#___________________________________________3 lab activity___________________________________________
# Supponendo che X e y siano definiti
criterion = 'gini'      # O "entropy"
# ccp_alpha = 0.01      # Parametro di potatura

# Trova il miglior ccp_alpha
# best_alpha = find_best_ccp_alpha(X, y)
# print(f"Il valore ottimale di ccp_alpha è: {best_alpha}")

# Costruzione dell'albero con criterio 'entropy' e senza pruning
T = decisionTreeLearner(X, y, c='entropy', ccp_alpha=0.0015)

# Visualizzazione e informazioni
showTree(T)

"""
    5.  Imparare un albero di decisione utilizzando l'intero corso e considerando la configurazione migliore 
    identificata con determineDecisionTreekFoldConfiguration.
"""
# Determina la configurazione migliore
best_criterion, best_ccp_alpha, best_f1 = determineDecisionTreekFoldConfiguration(X, y)
print(f"\nConfigurazione migliore: criterio = {best_criterion}, ccp_alpha = {best_ccp_alpha}")

# Addestra l'albero decisionale con la configurazione ottimale
T_best = decisionTreeLearner(X, y, c=best_criterion, ccp_alpha=best_ccp_alpha)

print("\n=======================================================================")



#___________________________________________4 lab activity___________________________________________
# load data tes
print("\n=======================================================================")
testpath="testDdosLabelNumeric.csv"

testData=load(testpath)
# ottieni la forma (numero di righe e colonne) del DataFrame e la stampa
shape=testData.shape
print("\nShape CSV test:")
print(shape)
# Pre-elaboration
# ottieni la lista delle colonne
cols = list(testData.columns.values)
preElaborationData(testData,cols)
# Drop useless columns in PANDAS
testData,removedColumns=removeColumns(testData,cols)
print("\nColonne rimosse:")
print(removedColumns)
print("=======================================================================")

# Caricamento e Valutazione del modello
evaluateDecisionTree(T_best, testData)

print("\n=======================================================================")
# 1) Determine the best Random Forest configuration
# Chiamata alla funzione per ottenere la configurazione migliore e il punteggio F1
criterion, randomization, bootstrap, n_trees, ccp_alpha, best_weighted_f1 = determineRFkFoldConfiguration(X, y)

# Stampa della configurazione migliore e del punteggio F1 ponderato
print(f"Migliore configurazione trovata:")
print(f"- Criterio: {criterion}")
print(f"- Randomizzazione: {randomization}")
print(f"- Bootstrap size: {bootstrap}")
print(f"- Numero di alberi: {n_trees}")
print(f"- ccp_alpha: {ccp_alpha}")
print(f"- Punteggio F1 ponderato medio: {best_weighted_f1}")

# 2) Train the Random Forest using the entire training set and the best configuration
# Crea il modello Random Forest utilizzando la configurazione migliore
rf_model = RandomForestClassifier(
    criterion=criterion,
    max_features=randomization,
    bootstrap=True,
    max_samples=bootstrap,
    n_estimators=n_trees,
    ccp_alpha=ccp_alpha,
    random_state=40         # Imposta un seed per la riproducibilità
)
# Allena il modello sull'intero set di addestramento
rf_model.fit(X, y)

# 3) Evaluate the Random Forest on the testing dataset
accuracy = testRandomForest(rf_model, testData)

print("\n=======================================================================")