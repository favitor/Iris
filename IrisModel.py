import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
 
classifier_model = skflow.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=iris.data.shape[1])],
                                    n_classes=3)

classifier_model.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target,classifier_model.predict(iris.data))

print("Accuracy: %f" % score)