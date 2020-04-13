# 检查版本
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# 导入数据、模块
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei']

# 训练集、验证集数据分割
seed = 1729140713
train,test,train_label,test_label = train_test_split(iris.data,iris.target,test_size = 0.3,random_state = seed)


# 选用机器学习算法
models = []
models.append(('决策树', DecisionTreeClassifier()))
models.append(('朴素贝叶斯', GaussianNB()))
models.append(('随机森林', RandomForestClassifier()))
models.append(('支持向量机SVM', SVC()))


# 基于test集的预测及验证
for name, model in models:
    model.fit(train, train_label)
    pre = model.predict(test)
    results = model.score(test, test_label)
    print("算法:{}\n准确率:{}{} ".format(name,results*100,"%"))
    print(classification_report(test_label,pre,target_names = iris.target_names))



# 交叉验证
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state = seed)
names=[]
scores=[]
for name, model in models:
    cfit = model.fit(X_train, Y_train)
    cfit.score(X_test, Y_test)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=10)
    scores.append(cv_scores)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_scores.mean(), cv_scores.std())
    print(msg)


# 算法比较
fig = plt.figure()
fig.suptitle('四种算法预测准确率比较图')
ax = fig.add_subplot(1,1,1)
plt.ylabel('算法')
plt.xlabel('准确率')
plt.boxplot(scores,vert=False,patch_artist=True,meanline=False,showmeans=True)
ax.set_yticklabels(names)
plt.show()
