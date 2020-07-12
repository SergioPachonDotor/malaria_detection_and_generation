#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



from sklearn.metrics import roc_curve, auc 
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def predict(model, test_imgs, batch_size):
	test_label_predict = model.predict(test_imgs, batch_size=batch_size)
	return test_label_predict

def plot_AUC_ROC(model, test_imgs, test_labels, model_name):
	loo = LeaveOneOut()
	print('### Cross validation ####')
	i = 0
	##cross validation
	for test_index, drop_index in loo.split(test_imgs, test_labels):
		if i%250 == 0 or i == len(test_imgs)-1:
			print('{} images done'.format(i))
		test_im, drop_test = test_imgs[test_index], test_imgs[drop_index]
		test_lab, drop_label = test_labels[test_index], test_labels[drop_index]
		i += 1


	test_label_predict = predict(model, test_im, batch_size=512)
	fpr, tpr, _ = roc_curve(test_lab, test_label_predict[:,0]) 
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(roc_auc),linewidth=2.5)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([-0.02, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc="lower right")
	plt.savefig('figures/'+model_name+'_AUC.png')
	plt.show()


##### CONFUSION METRICS #######

def get_metrics(true_labels, predicted_labels): 
	cm = metrics.confusion_matrix(true_labels, predicted_labels)
	tp = cm[0,0]
	tn = cm[1,1]
	fp = cm[0,1]
	fn = cm[1,0]

	sensitivity = tp/(tp+fn)
	specificity = tn/(tn+fp)
	return {
        	'Accuracy': np.round(metrics.accuracy_score(true_labels, predicted_labels), 4),
        	'Precision': np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'),4),
        	'Recall': np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 4),
        	'F1 Score': np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4),
        	'Sensivity': np.round(sensitivity,4),
        	'Specificity': np.round(specificity,4)
        	}

def evaluation_metrics(model, test_imgs, test_labels):
	test_label_predict = predict(model, test_imgs , batch_size=512)
	test_label_predict[:] = [np.where(test_label_predict[i] > 0.5, 1, 0) for i in range(len(test_label_predict))]

	model_metrics = get_metrics(true_labels= test_labels, predicted_labels=test_label_predict)
	result = pd.DataFrame([model_metrics], index=['Basic CNN']) 
	print(result)






