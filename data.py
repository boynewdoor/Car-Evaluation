import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
import os

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

# Apply Label Encoding for 'buying' and 'safety' (ordinal variables)
#label_encoder = LabelEncoder()
#df['buying'] = label_encoder.fit_transform(df['buying'])
#df['maint'] = label_encoder.fit_transform(df['safety'])

# Apply One-Hot Encoding for 'maint', 'doors', 'persons', 'lug_boot' (categorical variables)
#df_encoded = pd.get_dummies(df, columns=['maint', 'doors', 'persons', 'lug_boot'], drop_first=True)

# Calculando a porcentagem de valores únicos para cada coluna
unique_percentages = {}
total_rows = len(df)

for column in df.columns:
    counts = df[column].value_counts()
    percentages = (counts / total_rows * 100).round(3).astype(str) + " %"
    unique_percentages[column] = pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage']).rename_axis(None)

# Imprimindo os resultados
for column, result_df in unique_percentages.items():
    print(f"Column: {column}")
    print(result_df)
    print("\n")

# 1. Descriptive Statistics
print("1. Descriptive Statistics:")
print(df.describe())
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
    
    plt.show()

# 4. Análise de Outliers
# - Box plots para cada atributo
for column in df.columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='class', y=column, data=df)
    plt.title(f'Boxplot for {column} vs Class')
    
    # Save the boxplot image in the boxplot folder
    plt.savefig(os.path.join(boxplot_folder, f'boxplot_{column}_vs_class.png'))
    
    plt.show()

# 5. Tratamento de Dados Ausentes
# - Verificando a presença de dados ausentes
missing_data = df.isnull().sum()
print("Missing Data:")
print(missing_data)

# 7. Cross-Tabulation

# CROSS-TABULATION -> BUYING x Others
cross_tab = pd.crosstab(df['buying'], df['class'])
print("\nCross-Tabulation for BUYING x CLASS:")
print(cross_tab)
print("\n")

cross_tab1 = pd.crosstab(df['buying'], df['maint'])
print("\nCross-Tabulation for BUYING x MAINT:")
print(cross_tab1)
print("\n")

cross_tab2 = pd.crosstab(df['buying'], df['doors'])
print("\nCross-Tabulation for BUYING x DOORS:")
print(cross_tab2)
print("\n")

cross_tab3 = pd.crosstab(df['buying'], df['persons'])
print("\nCross-Tabulation for BUYING x PERSONS:")
print(cross_tab3)
print("\n")

cross_tab4 = pd.crosstab(df['buying'], df['lug_boot'])
print("\nCross-Tabulation for BUYING x LUG_BOOT:")
print(cross_tab4)
print("\n")

cross_tab5 = pd.crosstab(df['buying'], df['safety'])
print("\nCross-Tabulation for BUYING x SAFETY:")
print(cross_tab5)
print("\n")

# CROSS-TABULATION -> MAINT x Others
cross_tab6 = pd.crosstab(df['maint'], df['class'])
print("\nCross-Tabulation for MAINT x CLASS:")
print(cross_tab6)
print("\n")

cross_tab7 = pd.crosstab(df['maint'], df['doors'])
print("\nCross-Tabulation for MAINT x DOORS:")
print(cross_tab7)
print("\n")

cross_tab8 = pd.crosstab(df['maint'], df['persons'])
print("\nCross-Tabulation for MAINT x PERSONS:")
print(cross_tab8)
print("\n")

cross_tab9 = pd.crosstab(df['maint'], df['lug_boot'])
print("\nCross-Tabulation for MAINT x LUG_BOOT:")
print(cross_tab9)
print("\n")

cross_tab10 = pd.crosstab(df['maint'], df['safety'])
print("\nCross-Tabulation for MAINT x SAFETY:")
print(cross_tab10)
print("\n")

# CROSS-TABULATION -> DOORS x Others
cross_tab11 = pd.crosstab(df['doors'], df['class'])
print("\nCross-Tabulation for DOORS x CLASS:")
print(cross_tab11)
print("\n")

cross_tab12 = pd.crosstab(df['doors'], df['persons'])
print("\nCross-Tabulation for DOORS x PERSONS:")
print(cross_tab12)
print("\n")

cross_tab13 = pd.crosstab(df['doors'], df['lug_boot'])
print("\nCross-Tabulation for DOORS x LUG_BOOT:")
print(cross_tab13)
print("\n")

cross_tab14 = pd.crosstab(df['doors'], df['safety'])
print("\nCross-Tabulation for DOORS x SAFETY:")
print(cross_tab14)
print("\n")

# CROSS-TABULATION -> PERSONS x Others
cross_tab15 = pd.crosstab(df['persons'], df['class'])
print("\nCross-Tabulation for PERSONS x CLASS:")
print(cross_tab15)
print("\n")

cross_tab16 = pd.crosstab(df['persons'], df['lug_boot'])
print("\nCross-Tabulation for PERSONS x LUG_BOOT:")
print(cross_tab16)
print("\n")

cross_tab17 = pd.crosstab(df['persons'], df['safety'])
print("\nCross-Tabulation for PERSONS x SAFETY:")
print(cross_tab17)
print("\n")

# CROSS-TABULATION -> LUG_BOOT x Others
cross_tab18 = pd.crosstab(df['lug_boot'], df['class'])
print("\nCross-Tabulation for LUG_BOOT x CLASS:")
print(cross_tab18)
print("\n")

cross_tab19 = pd.crosstab(df['lug_boot'], df['safety'])
print("\nCross-Tabulation for LUG_BOOT x SAFETY:")
print(cross_tab19)
print("\n")

# CROSS-TABULATION -> SAFETY x Others
cross_tab20 = pd.crosstab(df['safety'], df['class'])
print("\nCross-Tabulation for SAFETY x CLASS:")
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

### 9. Predictive Modelling -> ACCURACY###

X = df.drop('class', axis=1)
y = df['class']

# Encode categorical variables
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = "{:.3%}".format(accuracy)
print(f'Accuracy: {accuracy}' + ' = ' + f'{accuracy_percentage}')
print("\n")