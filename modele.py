from dataset import *

from sklearn.base import BaseEstimator
import seaborn as sns
import numpy as np 


class MyEstimator(BaseEstimator):
    
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.prediction = None
        self.score = None
        self.param = None
        self.anomalie = None
        self.X = None
        self.name = None
        

    def fit(self, X):
        self.X = X
        print(str(self.base_estimator)[0:6])
        if  str(self.base_estimator)[0:6] ==  "LocalO" or str(self.base_estimator)[0:6] == "Gaussi": 
            self.prediction = self.base_estimator.fit_predict(X)
        else : 
            self.base_estimator.fit(X)
        return self.base_estimator


    def predict(self):
        if  str(self.base_estimator)[0:6] ==  "LocalO" :
            self.name = "Local outlier factor"
            self.score = self.base_estimator.negative_outlier_factor_
            
        elif str(self.base_estimator)[0:6] ==  "DBSCAN" :
            self.name = "DBSCAN"
            label = self.base_estimator.labels_
            label= np.where(label != -1, 1,-1)
            self.prediction = label
            
            
        elif str(self.base_estimator)[0:6] == "Gaussi" :
            self.name = "Gaussian Mixture modele"
            self.score = self.base_estimator.score_samples(self.X)
            pct_threshold = np.percentile(self.score, 3)
            pred = pd.DataFrame()
            pred["score"] = self.score 
            self.prediction= pred["score"].apply(lambda x: 1 if x > pct_threshold else -1)
            
        else : 
            self.prediction = self.base_estimator.predict(self.X)
            self.score = self.base_estimator.score_samples(self.X)
            self.name = "Isolation forest"
       
        self.param = self.base_estimator.get_params(deep=False)
        return 
    

    def evaluation(self) :
        anomalieP = pd.DataFrame(self.prediction).value_counts(normalize=True).mul(100).round(1).astype(str)
        titre = self.name + " avec " + str(anomalieP[-1]) + " % Anomalie " 
        
        
        fig, axes = plt.subplots(2, 1)
        fig.suptitle(titre, fontsize=16)
        
        sns.lineplot(x=self.X.index, y = self.X["value"],color="blue",ax=axes[0])
        sns.scatterplot(x=np.array(self.X.index) , y = np.array(self.X["value"]), hue=self.prediction, palette=["r", "b"],markers=True,ax=axes[0])
     
        sns.histplot(data=np.array(self.X.index), x=np.array(self.X["value"]), bins=20, hue=self.prediction, palette=['lightgreen', 'pink'],ax=axes[1])
        axes[1].legend(labels=["Non anomalie","Anomalie"])
        
        
        chemin = "Result/" + str(self.base_estimator) + ".png"

        plt.savefig(chemin)
        plt.show()
       
       










 
 