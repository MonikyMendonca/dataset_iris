from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

def visualizar_dados(X, y, feature_names, target_names):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = df['target'].apply(lambda x: target_names[x])
    
    sns.pairplot(df, hue='target_name')
    plt.suptitle("Visualização das Flores Iris", y=1.02)
    plt.show()

def testar_valores_de_k(X_train, X_test, y_train, y_test):
    print("Testando diferentes valores de k:")
    for k in range(1, 11):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"k = {k}: accuracy = {score:.2f}")

def main():
   
    iris = load_iris()
    X = iris.data
    y = iris.target

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    visualizar_dados(X, y, iris.feature_names, iris.target_names)

    
    testar_valores_de_k(X_train, X_test, y_train, y_test)

    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    
    joblib.dump(model, "modelo_knn_iris.pkl")
    print("\nModelo salvo como 'modelo_knn_iris.pkl'.")

if __name__ == "__main__":
    main()
