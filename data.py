import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay 
from scipy.stats import chi2_contingency
import os
import numpy as np
from io import StringIO
from tabulate import tabulate

plt.rcParams['figure.max_open_warning'] = 50  # Set the threshold to 50 or a value suitable for your case

# Carregando o arquivo de dados
data_file = "car.data"
data_df = pd.read_csv(data_file, header=None)

# Definindo os atributos desejados na ordem fornecida
desired_attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Criando um DataFrame com os atributos como cabeçalhos e os dados de data_df
combined_df = pd.DataFrame(data_df.values, columns=desired_attributes)

# Salvando o DataFrame em um arquivo CSV sem incluir índices
combined_df.to_csv("data.csv", index=False)

# Imprimindo os dados originais com os headers desejados
print("\nOriginal Data:")
print(data_df.rename(columns=dict(enumerate(desired_attributes))))
print("\n")

# Carregando o arquivo CSV
file_path = "data.csv"
df = pd.read_csv(file_path)

unique_percentages_folder = "unique_percentages"
os.makedirs(unique_percentages_folder, exist_ok=True)

# Calculando a porcentagem de valores únicos para cada coluna
unique_percentages = {}
total_rows = len(df)

for column in df.columns:
    counts = df[column].value_counts()
    percentages = (counts / total_rows * 100).round(3).astype(str) + " %"
    unique_percentages[column] = pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage']).rename_axis(None)

# Remove the 'class' header
    if 'class' in unique_percentages[column].index:
        unique_percentages[column].index = ['' if idx == 'class' else idx for idx in unique_percentages[column].index]

# Salvando tabelas como imagens .png
for column, result_df in unique_percentages.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.table(cellText=result_df.values, colLabels=result_df.columns, rowLabels=result_df.index, cellLoc='center', loc='center')
    plt.savefig(os.path.join(unique_percentages_folder, f'tabela_{column}_unique_percentages.png'))
    plt.close()
    print(f"Column: {column}")
    print(result_df)
    print("\n")

# 1. Descriptive Statistics
print("1. Descriptive Statistics:")
descriptive_statistics_folder = "descriptive_statistics"
os.makedirs(descriptive_statistics_folder, exist_ok=True)
desc_stats = df.describe()
# Adiciona uma coluna antes de cada coluna nas estatísticas descritivas
# Adiciona uma coluna com um cabeçalho aleatório antes da coluna 'buying'
# Cria um cabeçalho aleatório
random_header = np.random.choice(['Random_Header'])

# Adiciona uma coluna com os valores 'count', 'unique', 'top' e 'freq'
desc_stats[random_header] = ['count', 'unique', 'top', 'freq']

# Reorganiza as colunas para que 'Random_Header' esteja antes de 'buying'
desc_stats = desc_stats[['Random_Header'] + [col for col in desc_stats.columns if col != 'Random_Header']]

# Remove o cabeçalho da coluna 'Random_Header'
desc_stats.columns = ['' if col == 'Random_Header' else col for col in desc_stats.columns]

table_desc_stats = desc_stats.to_html
# Salvar tabela HTML como imagem .png
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.table(cellText=desc_stats.values, colLabels=desc_stats.columns, cellLoc='center', loc='center')
plt.savefig(os.path.join(descriptive_statistics_folder, 'tabela_desc_stats.png'))
plt.close()
print(desc_stats)
print("\n")

# Create folders to save the images
countplot_folder = "countplot"
boxplot_folder = "boxplot"

os.makedirs(countplot_folder, exist_ok=True)
os.makedirs(boxplot_folder, exist_ok=True)

# 2. Visualização de Dados
# - Gráficos de contagem para variáveis categóricas
for column in df.columns[:-1]:  # Excluir a última coluna (classe) para variáveis categóricas
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column, data=df, hue='class')
    plt.title(f'Countplot for {column} vs Class')
    
    # Save the countplot image in the countplot folder
    plt.savefig(os.path.join(countplot_folder, f'countplot_{column}_vs_class.png'))
    plt.close()

# 4. Análise de Outliers
# - Box plots para cada atributo
for column in df.columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='class', y=column, data=df)
    plt.title(f'Boxplot for {column} vs Class')
    
    # Save the boxplot image in the boxplot folder
    plt.savefig(os.path.join(boxplot_folder, f'boxplot_{column}_vs_class.png'))
    plt.close()

# 5. Tratamento de Dados Ausentes
# - Verificando a presença de dados ausentes
missing_data = df.isnull().sum()
print("Missing Data:")
print(missing_data)
print("\n")

# 7. Cross-Tabulation

# CROSS-TABULATION -> BUYING x Others
cross_tab = pd.crosstab(df['buying'], df['class'])
print(cross_tab)
print("\n")

cross_tab1 = pd.crosstab(df['buying'], df['maint'])
print(cross_tab1)
print("\n")

cross_tab2 = pd.crosstab(df['buying'], df['doors'])
print(cross_tab2)
print("\n")

cross_tab3 = pd.crosstab(df['buying'], df['persons'])
print(cross_tab3)
print("\n")

cross_tab4 = pd.crosstab(df['buying'], df['lug_boot'])
print(cross_tab4)
print("\n")

cross_tab5 = pd.crosstab(df['buying'], df['safety'])
print(cross_tab5)
print("\n")

# CROSS-TABULATION -> MAINT x Others
cross_tab6 = pd.crosstab(df['maint'], df['class'])
print(cross_tab6)
print("\n")

cross_tab7 = pd.crosstab(df['maint'], df['doors'])
print(cross_tab7)
print("\n")

cross_tab8 = pd.crosstab(df['maint'], df['persons'])
print(cross_tab8)
print("\n")

cross_tab9 = pd.crosstab(df['maint'], df['lug_boot'])
print(cross_tab9)
print("\n")

cross_tab10 = pd.crosstab(df['maint'], df['safety'])
print(cross_tab10)
print("\n")

# CROSS-TABULATION -> DOORS x Others
cross_tab11 = pd.crosstab(df['doors'], df['class'])
print(cross_tab11)
print("\n")

cross_tab12 = pd.crosstab(df['doors'], df['persons'])
print(cross_tab12)
print("\n")

cross_tab13 = pd.crosstab(df['doors'], df['lug_boot'])
print(cross_tab13)
print("\n")

cross_tab14 = pd.crosstab(df['doors'], df['safety'])
print(cross_tab14)
print("\n")

# CROSS-TABULATION -> PERSONS x Others
cross_tab15 = pd.crosstab(df['persons'], df['class'])
print(cross_tab15)
print("\n")

cross_tab16 = pd.crosstab(df['persons'], df['lug_boot'])
print(cross_tab16)
print("\n")

cross_tab17 = pd.crosstab(df['persons'], df['safety'])
print(cross_tab17)
print("\n")

# CROSS-TABULATION -> LUG_BOOT x Others
cross_tab18 = pd.crosstab(df['lug_boot'], df['class'])
print(cross_tab18)
print("\n")

cross_tab19 = pd.crosstab(df['lug_boot'], df['safety'])
print(cross_tab19)
print("\n")

# CROSS-TABULATION -> SAFETY x Others
cross_tab20 = pd.crosstab(df['safety'], df['class'])
print(cross_tab20)
print("\n")

# 8. Chi-Square Test

# CHI-SQUARE -> BUYING
chi2, p, dof, expected = chi2_contingency(cross_tab)
print("\nChi-Square Test for BUYING x CLASS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab1)
print("\nChi-Square Test for BUYING x MAINT:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab2)
print("\nChi-Square Test for BUYING x DOORS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab3)
print("\nChi-Square Test for BUYING x PERSONS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab4)
print("\nChi-Square Test for BUYING x LUG_BOOT:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab5)
print("\nChi-Square Test for BUYING x SAFETY:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

# CHI-SQUARE -> MAINT
chi2, p, dof, expected = chi2_contingency(cross_tab6)
print("\nChi-Square Test for MAINT x CLASS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab7)
print("\nChi-Square Test for MAINT x DOORS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab8)
print("\nChi-Square Test for MAINT x PERSONS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab9)
print("\nChi-Square Test for MAINT x LUG_BOOT:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab10)
print("\nChi-Square Test for MAINT x SAFETY:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

# CHI-SQUARE -> DOORS
chi2, p, dof, expected = chi2_contingency(cross_tab11)
print("\nChi-Square Test for DOORS x CLASS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab12)
print("\nChi-Square Test for DOORS x PERSONS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab13)
print("\nChi-Square Test for DOORS x LUG_BOOT:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab14)
print("\nChi-Square Test for DOORS x SAFETY:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

# CHI-SQUARE -> PERSONS
chi2, p, dof, expected = chi2_contingency(cross_tab15)
print("\nChi-Square Test for PERSONS x CLASS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab16)
print("\nChi-Square Test for PERSONS x LUG_BOOT:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab17)
print("\nChi-Square Test for PERSONS x SAFETY:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

# CHI-SQUARE -> LUG_BOOT
chi2, p, dof, expected = chi2_contingency(cross_tab18)
print("\nChi-Square Test for LUG_BOOT x CLASS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

chi2, p, dof, expected = chi2_contingency(cross_tab19)
print("\nChi-Square Test for LUG_BOOT x SAFETY:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

# CHI-SQUARE -> SAFETY
chi2, p, dof, expected = chi2_contingency(cross_tab20)
print("\nChi-Square Test for SAFETY x CLASS:")
print(f"Chi2: {chi2}, p-value: {p}")
print("\n")

### 9. Predictive Modelling ###

models_folder = "models"

os.makedirs(models_folder, exist_ok=True)

X = df.drop('class', axis=1)
y = df['class']

# Encode categorical variables
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

### RANDOM FOREST ###

print("### RANDOM FOREST ###")

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred).round(3)
accuracy_percentage_rf = "{:.3%}".format(accuracy_rf)
print(f'Accuracy: {accuracy_rf}' + ' = ' + f'{accuracy_percentage_rf}')
print("\n")

# Avaliação do modelo de Random Forest
print("Classification Report:")
classification_report_str = classification_report(y_test, y_pred)
print(classification_report_str)

# AUC-ROC para um problema de classificação multiclasse
y_prob = rf_classifier.predict_proba(X_test)  # Probabilidades de classe
auc_roc_rf = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr').round(3)
print("AUC-ROC:", auc_roc_rf)
print("\n")

# Converter o relatório de classificação em DataFrame para facilitar a manipulação
classification_report_df = pd.read_fwf(StringIO(classification_report_str), index_col=0)
# Adicionar a coluna com os valores especificados no início da tabela
classification_report_df.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
# Remover o cabeçalho da coluna 'Header_Column'
classification_report_df.columns = ['' if col == 'Header_Column' else col for col in classification_report_df.columns]
# Substituir "nan" por uma string vazia
classification_report_df = classification_report_df.fillna('')

# Criar tabela HTML para o relatório de classificação
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.table(cellText=classification_report_df.values, colLabels=classification_report_df.columns, cellLoc='center', loc='center')
plt.savefig(os.path.join(models_folder, 'tabela_classification_report_random_forest.png'))
plt.close()

### SVM - SUPPORT VECTOR MACHINE ###

print("\n### SVM - SUPPORT VECTOR MACHINE ###")

# Initialize and train a Support Vector Machine classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred).round(3)
accuracy_percentage_svm = "{:.3%}".format(accuracy_svm)
print(f'Accuracy: {accuracy_svm}' + ' = ' + f'{accuracy_percentage_svm}')
print("\n")

# Avaliação do modelo de Support Vector Machine
print("Classification Report:")
classification_report_str1 = classification_report(y_test, y_pred)
print(classification_report_str1)

# AUC-ROC para um problema de classificação multiclasse
y_prob = svm_classifier.decision_function(X_test)  # Função de decisão para probabilidades
auc_roc_svm = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr').round(3)
print("AUC-ROC:", auc_roc_svm)
print("\n")

# Converter o relatório de classificação em DataFrame para facilitar a manipulação
classification_report_df1 = pd.read_fwf(StringIO(classification_report_str1), index_col=0)
# Adicionar a coluna com os valores especificados no início da tabela
classification_report_df1.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
# Remover o cabeçalho da coluna 'Header_Column'
classification_report_df1.columns = ['' if col == 'Header_Column' else col for col in classification_report_df1.columns]
# Substituir "nan" por uma string vazia
classification_report_df1 = classification_report_df1.fillna('')

# Criar tabela HTML para o relatório de classificação
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.table(cellText=classification_report_df1.values, colLabels=classification_report_df1.columns, cellLoc='center', loc='center')
plt.savefig(os.path.join(models_folder, 'tabela_classification_report_svm.png'))
plt.close()

### NEURAL NETWORKS ###

print("\n### NEURAL NETWORKS ###")

# Inicializando o modelo MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Treinando o modelo
mlp_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred_mlp = mlp_model.predict(X_test)

# Avaliando o modelo
accuracy_mlp = accuracy_score(y_test, y_pred_mlp).round(3)
accuracy_percentage_mlp = "{:.3%}".format(accuracy_mlp)
print(f'Accuracy (MLP): {accuracy_mlp}' + ' = ' + f'{accuracy_percentage_mlp}')
print("\n")

# Classification Report
print("Classification Report (MLP):")
classification_report_str2 = classification_report(y_test, y_pred_mlp)
print(classification_report_str2)

# AUC-ROC
y_prob_mlp = mlp_model.predict_proba(X_test)  # Probabilidades de classe
auc_roc_mlp = roc_auc_score(pd.get_dummies(y_test), y_prob_mlp, multi_class='ovr').round(3)
print("AUC-ROC (MLP):", auc_roc_mlp)
print("\n")

# Converter o relatório de classificação em DataFrame para facilitar a manipulação
classification_report_df2 = pd.read_fwf(StringIO(classification_report_str2), index_col=0)
# Adicionar a coluna com os valores especificados no início da tabela
classification_report_df2.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
# Remover o cabeçalho da coluna 'Header_Column'
classification_report_df2.columns = ['' if col == 'Header_Column' else col for col in classification_report_df2.columns]
# Substituir "nan" por uma string vazia
classification_report_df2 = classification_report_df2.fillna('')

# Criar tabela HTML para o relatório de classificação
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.table(cellText=classification_report_df2.values, colLabels=classification_report_df2.columns, cellLoc='center', loc='center')
plt.savefig(os.path.join(models_folder, 'tabela_classification_report_neural_networks.png'))
plt.close()

### DECISION TREES ### 

print("\n### DECISION TREES ###")

# Inicializando o modelo de Árvore de Decisão
dt_model = DecisionTreeClassifier(random_state=42)

# Treinando o modelo
dt_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred_dt = dt_model.predict(X_test)

# Avaliando o modelo
accuracy_dt = accuracy_score(y_test, y_pred_dt).round(3)
accuracy_percentage_dt = "{:.3%}".format(accuracy_dt)
print(f'Accuracy (Decision Tree): {accuracy_dt}' + ' = ' + f'{accuracy_percentage_dt}')
print("\n")

# Classification Report
print("Classification Report (Decision Tree):")
classification_report_str3 = classification_report(y_test, y_pred_dt)
print(classification_report_str3)
print("\n")

# Converter o relatório de classificação em DataFrame para facilitar a manipulação
classification_report_df3 = pd.read_fwf(StringIO(classification_report_str3), index_col=0)
# Adicionar a coluna com os valores especificados no início da tabela
classification_report_df3.insert(0, 'Header_Column', ['acc', 'good', 'unacc', 'vgood', 'accuracy', 'macro avg', 'weighted avg'])
# Remover o cabeçalho da coluna 'Header_Column'
classification_report_df3.columns = ['' if col == 'Header_Column' else col for col in classification_report_df3.columns]
# Substituir "nan" por uma string vazia
classification_report_df3 = classification_report_df3.fillna('')

# Criar tabela HTML para o relatório de classificação
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.table(cellText=classification_report_df3.values, colLabels=classification_report_df3.columns, cellLoc='center', loc='center')
plt.savefig(os.path.join(models_folder, 'tabela_classification_report_decision_trees.png'))
plt.close()

# table of models with precisions

model_data = [
    ("RANDOM FOREST", accuracy_rf, accuracy_percentage_rf, auc_roc_rf),
    ("SVM", accuracy_svm, accuracy_percentage_svm, auc_roc_svm),
    ("NEURAL NETWORKS", accuracy_mlp, accuracy_percentage_mlp, auc_roc_mlp),
    ("DECISION TREE", accuracy_dt, accuracy_percentage_dt),
]

# Define headers for the table
headers = ["Model", "Precision Score", "Percentage", "AUC-ROC"]

# Create the table using tabulate
table = tabulate(model_data, headers=headers, tablefmt="pipe")

print("### MODELS TABLES ###\n")

# Print the table
print(table)

# Compare precision scores
best_model_precision = max(model_data, key=lambda x: x[1])
print(f"\nThe best model based on precision score is {best_model_precision[0]} with a precision score of {best_model_precision[1]:.2f} corresponding to a percentage of {best_model_precision[2]}%.")

# Compare AUC-ROC scores
try:
    best_model_auc_roc = max((model for model in model_data if len(model) == len(headers)), key=lambda x: x[3])
    print(f"\nThe best model based on AUC-ROC score is {best_model_auc_roc[0]} with an AUC-ROC score of {best_model_auc_roc[3]:.2f}.\n")
except ValueError:
    print("Error: AUC-ROC values are not available for any model.")

# Confusion Matrix
    
# Create a directory to save confusion matrixes if it doesn't exist
output_directory = 'confusion_matrixes'
os.makedirs(output_directory, exist_ok=True)
    
print("CONFUSION MATRIXES generating ...\n")

models = {
    'Random Forest': rf_classifier,
    'SVM': svm_classifier,
    'Decision Tree': dt_model,
    'Neural Network': mlp_model
}

for model_name, model in models.items():
    # train the model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    #calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix
    print(f'### CONFUSION MATRIX - {model_name} ###\n')
    print("\n")
    print(cm)
    print("\n")

    #Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.savefig(os.path.join(output_directory, f'{model_name}_confusion_matrix.png'))
    plt.close()  # Close the plot to avoid displaying in the notebook