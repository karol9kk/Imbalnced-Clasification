from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
import pandas as pd
import numpy as np

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def kmeans(df):
    kmeans = KMeans(n_clusters=2,random_state=42)
    kmeans.fit(df)
    clusters = kmeans.predict(df)
    df["Cluster"] = clusters

    return df

def get_plotX_df(df):

    df=kmeans(df)
    
    plotX = pd.DataFrame(np.array(df.sample(100000)))

    #Rename plotX's columns since it was briefly converted to an np.array above
    plotX.columns = df.columns

    #PCA with one principal component
    pca_1d = PCA(n_components=1)

    #PCA with two principal components
    pca_2d = PCA(n_components=2)

    #PCA with three principal components
    pca_3d = PCA(n_components=3)

    #This DataFrame holds that single principal component mentioned above
    PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

    #This DataFrame contains the two principal components that will be used
    #for the 2-D visualization mentioned above
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

    #And this DataFrame contains three principal components that will aid us
    #in visualizing our clusters in 3-D
    PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

    PCs_1d.columns = ["PC1_1d"]

    #"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
    #And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]

    PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

    plotX = pd.concat([plotX,PCs_1d,PCs_2d,PCs_3d], axis=1, join='inner')

    plotX["dummy"] = 0

    return plotX

def cluster_1d(df):

    plotX=get_plotX_df(df)
    
    init_notebook_mode(connected=True)
    
    cluster0 = plotX[plotX["Cluster"] == 0]
    cluster1 = plotX[plotX["Cluster"] == 1] 

    trace1 = go.Scatter(
                    x = cluster0["PC1_1d"],
                    y = cluster0["dummy"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

    #trace2 is for 'Cluster 1'
    trace2 = go.Scatter(
                        x = cluster1["PC1_1d"],
                        y = cluster1["dummy"],
                        mode = "markers",
                        name = "Cluster 1",
                        marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                        text = None)

    

    data = [trace1, trace2,]

    title = "Visualizing Clusters in One Dimension Using PCA"

    layout = dict(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= '',ticklen= 5,zeroline= False)
                )

    fig = dict(data = data, layout = layout)

    iplot(fig)

def cluster_2d(df):
    
    plotX=get_plotX_df(df)
    
    init_notebook_mode(connected=True)
    
    cluster0 = plotX[plotX["Cluster"] == 0]
    cluster1 = plotX[plotX["Cluster"] == 1] 
    
    #trace1 is for 'Cluster 0'
    trace1 = go.Scatter(
                        x = cluster0["PC1_2d"],
                        y = cluster0["PC2_2d"],
                        mode = "markers",
                        name = "Cluster 0",
                        marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                        text = None)

    #trace2 is for 'Cluster 1'
    trace2 = go.Scatter(
                        x = cluster1["PC1_2d"],
                        y = cluster1["PC2_2d"],
                        mode = "markers",
                        name = "Cluster 1",
                        marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                        text = None)

    

    data = [trace1, trace2]

    title = "Visualizing Clusters in Two Dimensions Using PCA"

    layout = dict(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                )

    fig = dict(data = data, layout = layout)

    iplot(fig)

def cluster_3d(df):

    plotX=get_plotX_df(df)
    
    init_notebook_mode(connected=True)
    
    cluster0 = plotX[plotX["Cluster"] == 0]
    cluster1 = plotX[plotX["Cluster"] == 1] 


    trace1 = go.Scatter3d(
                    x = cluster0["PC1_3d"],
                    y = cluster0["PC2_3d"],
                    z = cluster0["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

    #trace2 is for 'Cluster 1'
    trace2 = go.Scatter3d(
                        x = cluster1["PC1_3d"],
                        y = cluster1["PC2_3d"],
                        z = cluster1["PC3_3d"],
                        mode = "markers",
                        name = "Cluster 1",
                        marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                        text = None)

    data = [trace1, trace2]

    title = "Visualizing Clusters in Three Dimensions Using PCA"

    layout = dict(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                )

    fig = dict(data = data, layout = layout)

    iplot(fig)