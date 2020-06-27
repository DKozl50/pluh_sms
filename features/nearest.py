from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def PredictCardHolder(data):
    scl = StandardScaler()
    svd = PCA(n_components=5)
    new_features = svd.fit_transform(scl.fit_transform(data.drop(['CardHolder', 'response_att', 'group'], axis=1)))
    
    knn = KNeighborsClassifier()
    knn.fit(new_features[np.where(data['group']==1)], data[data['group']==1]['CardHolder'])
    predicted_class = knn.predict(new_features[np.where(data['group']==0)])  
    pair_ch = np.hstack([data[data['group']==0]['CardHolder'].values.reshape(-1, 1), predicted_class.reshape(-1, 1)])
    answer = pd.DataFrame(pair_ch, columns=['CardHoldet', 'Predicted'])
    return answer