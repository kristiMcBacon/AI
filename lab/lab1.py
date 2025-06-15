#IMPORT LIBRERIE--------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


#-----------------------------------------------------
#Load data with PANDAS.read_csv     
def load(trainpath):
    """
        Carica i dati da un file CSV e restituisce un DataFrame.

        :param trainpath: Percorso del file CSV da caricare.
        :return: DataFrame con i dati caricati dal file CSV.
    """ 
    df = pd.read_csv(trainpath)
    return df


#-----------------------------------------------------
#Pre-elaborate data with PANDAS.DataFrame.describe
def preElaborationData(data, cols):
    """
        Questo metodo fornisce un riepilogo statistico delle variabili nel DataFrame, includendo:
        - count: il numero di valori non nulli.
        - mean: la media dei valori.
        - std: la deviazione standard dei valori.
        - min: il valore minimo.
        - 25%: il primo quartile (Q1).
        - 50%: la mediana (Q2).
        - 75%: il terzo quartile (Q3).
        - max: il valore massimo.

        Pre-elabora i dati stampando una descrizione statistica per ogni variabile.
        
        :param data: DataFrame contenente i dati.
        :param cols: Lista dei nomi delle colonne (variabili indipendenti).
    """
    # Stampa una descrizione di ogni variabile
    print("\n=======================================================================")
    print("Pre-elaborate data with PANDAS.DataFrame.describe:")

    print("--------------------------------------------------")    

    # Per ogni colonna nella lista delle colonne, calcola e stampa la descrizione
    for col in cols:
        print(f"Descrizione della colonna '{col}':")
        print(data[col].describe())  # Descrizione statistica della colonna
        print("--------------------------------------------------")

    # Analizza la distribuzione delle variabili indipendenti
    #for col in cols:
    #    if data[col].dtype in ['int64', 'float64']:  # Controlla solo le variabili numeriche
    #        print(f"\nDistribuzione di {col}:")
    #        print(data[col].value_counts(normalize=True).sort_index())  # Percentuale di ogni valore
    #        print(data[col].hist(bins=30))  # Istogramma della variabile
    print("=======================================================================")


#-----------------------------------------------------
# Drop useless columns in PANDAS
def removeColumns(data, cols):
    """
    Rimuove le variabili inutili dal DataFrame in base alla loro distribuzione statistica.
    
    :param data: DataFrame contenente i dati.
    :param cols: Lista dei nomi delle colonne (variabili indipendenti).
    :return: DataFrame con le colonne rimosse e lista delle colonne rimosse.
    """
    removed_columns = []  # Lista per tenere traccia delle colonne rimosse
    
    # Itera su ogni colonna e analizza la distribuzione
    for col in cols:
        desc = data[col].describe()  # Ottieni le statistiche descrittive della colonna
        
        # Se la deviazione standard è 0, significa che tutti i valori sono uguali (colonna costante)
        if desc['std'] == 0:
            removed_columns.append(col)  # Aggiungi alla lista delle colonne rimosse
            data = data.drop(columns=[col])  # Rimuovi la colonna dal DataFrame
            continue
        
        # Se la colonna ha pochi valori distinti (ad esempio, meno di 5), potrebbe non essere utile
        if len(data[col].unique()) < 5:
            removed_columns.append(col)  # Aggiungi alla lista delle colonne rimosse
            data = data.drop(columns=[col])  # Rimuovi la colonna dal DataFrame

    return data, removed_columns


#-----------------------------------------------------
# Funzione per la pre-elaborazione e per tracciare l'istogramma della distribuzione della classe
def preElaborationClass(data, target_column):
    """
    Mostra l'istogramma della distribuzione dei valori della classe (target variable).
    
    :param data: DataFrame contenente i dati.
    :param target_column: Nome della colonna target (ad esempio 'Label').
    """
    # Calcola la distribuzione della colonna target
    class_distribution = data[target_column].value_counts().sort_index()
    
    # Crea il grafico
    fig, ax = plt.subplots(figsize=(7, 4))  # Imposta la dimensione dell'immagine
    data[target_column].hist(bins=5, ax=ax)  # Crea l'istogramma con 5 intervalli
    
    # Aggiungi spazio sulla sinistra per separare il testo e il grafico
    plt.subplots_adjust(left=0.35) 
    
    # Posiziona il testo della distribuzione sulla sinistra del grafico
    y_position = 0.7        # Posizione da cui si inizia a stampare i valori
    fig.text(0.05, y_position, f"{target_column}", fontsize=10, verticalalignment='top')
    y_position -= 0.05      # Sposta il testo più in basso per ogni riga
    #crea figura con i valori
    for i, (value, count) in enumerate(class_distribution.items()):
        fig.text(0.05, y_position, f"{value}         {count}", fontsize=10, verticalalignment='top')
        y_position -= 0.05  # Sposta il testo più in basso per ogni riga
    
    # Titolo e etichette degli assi
    ax.set_title(f'Histogram of "{target_column}"')  # Titolo del grafico
    #ax.set_xlabel(target_column)  # Etichetta sull'asse x
    ax.set_ylabel('frequency')  # Etichetta sull'asse y
    ax.grid(False)  # Mostra la griglia
    
    # Visualizza il grafico
    plt.show()