from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class pca: 
    def train(self,df): 
        # Separating out the features
        features = []
        x = df.loc[:, features].values
        # Separating out the target
        y = df.loc[:,['target']].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents)
        finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
    
    def visualize_2d(self, finalDf): 
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , s = 50)
        ax.legend(targets)
        ax.grid()

    def var(self): 
        pca.explained_variance_ratio_