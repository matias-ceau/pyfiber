from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FiberPCA:
    '''Input must be a dataframe with subjects as rows and parameters as columns, additional parameters include number of components
    (no limit but plots and heatmaps can only show 2 to 3).'''
    
    def __init__(self,df,n=3,colormap=None):
        self.dim = n
        self.data = df #untransformed data
        self.scaler = StandardScaler() ; self.scaler.fit(df) #scaler
        self.scaled_data = self.scaler.transform(df) #normalised data as numpy array
        self.scaled_df = pd.DataFrame(self.scaled_data,index=df.index,columns=df.columns) #normalised data as dataframe
        self.pca = PCA(n_components=n) ; self.pca.fit(self.scaled_data) #pca
        self.x_pca = self.pca.transform(self.scaled_data) #pca results as array
        self.d_pca = pd.DataFrame(self.x_pca,index=df.index,columns=['PC' + str(i+1) for i in range(n)])
        self.components = pd.DataFrame(self.pca.components_,columns=df.columns,index=['PC' + str(i+1) for i in range(n)])
        if colormap:
            try:
                self.colormap = np.array(self.scaled_df[colormap])
                self.colormap[self.colormap >= np.median(self.colormap)] = 1
                self.colormap[self.colormap != 1] = 0
            except:
                self.colormap = None ; print('Not a label')
        else: self.colormap = None
        
    def plot(self,dim=2,labels=False,color='red',figsize=None):
        if dim == 2:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(xlabel='PC1',ylabel='PC2')
            ax.scatter(self.x_pca[:,0],self.x_pca[:,1],c=self.colormap)
        if dim == 3:
            plt.figure(figsize=figsize)
            ax = plt.axes(projection='3d',xlabel='PC1',ylabel='PC2',zlabel='PC3')
            ax.scatter3D(self.x_pca[:,0],self.x_pca[:,1],self.x_pca[:,2])
            if labels:
                for i, txt in enumerate(self.data.index):
                    ax.text(self.x_pca[:,0][i], self.x_pca[:,1][i], self.x_pca[:,2][i], txt, color=color)
                    
    def heatmap(self,save=False,minv='auto',maxv='auto',colormap='PiYG',figsize=(20,5)):
        plt.figure(figsize=figsize)
        if minv == 'auto': minv = np.array(self.components).min()
        if maxv == 'auto': maxv = np.array(self.components).max()
        if self.dim == 3:
            vpc1,vpc2,vpc3 = self.pca.explained_variance_ratio_
            sns.heatmap(self.components,vmin=minv,vmax=maxv,cmap=colormap,yticklabels=[f'PC1 {round(100*vpc1,1)}%',f'PC2 {round(100*vpc2,1)}%',f'PC3 {round(100*vpc3,1)}%'])
        if self.dim == 2:
            vpc1,vpc2 = self.pca.explained_variance_ratio_
            sns.heatmap(self.components,vmin=minv,vmax=maxv,cmap=colormap,yticklabels=[f'PC1 {round(100*vpc1,1)}%',f'PC2 {round(100*vpc2,1)}%'])
        if save: plt.savefig('pca.png')
    
    def showPC(self,a):
        if 0 < a <= self.dim:
            return pd.DataFrame(self.components.transpose()[f'PC{a}']).sort_values(f'PC{a}',ascending=False,key=abs)