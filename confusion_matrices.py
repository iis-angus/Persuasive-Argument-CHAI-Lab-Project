from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

#Load data
explain_then_predict_data = pd.read_csv('results/explain_then_predict_results.csv')
predict_then_explain_data = pd.read_csv('results/predict_then_explain_results.csv')

def explain_conf_matrix():
    cm_explain = confusion_matrix(explain_then_predict_data['correct_response'], explain_then_predict_data['prediction'], labels=[1, 2])
    accuracy_explain = accuracy_score(explain_then_predict_data['correct_response'], explain_then_predict_data['prediction'])
    display_explain = ConfusionMatrixDisplay(cm_explain, display_labels=['Response 1', 'Response 2'])

    #Displaying matrix
    plt.figure(figsize=(8, 6))
    display_explain.plot(cmap='Blues')
    plt.title(f'Explain-then-Predict Confusion Matrix\nAccuracy: {accuracy_explain:.3f}')
    plt.savefig('results/explain_then_predict_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_conf_matrix():
    cm_predict = confusion_matrix(predict_then_explain_data['correct_response'], predict_then_explain_data['prediction'], labels=[1, 2])
    accuracy_predict = accuracy_score(predict_then_explain_data['correct_response'], predict_then_explain_data['prediction'])
    display_predict = ConfusionMatrixDisplay(cm_predict, display_labels=['Response 1', 'Response 2'])

    #Displaying matrix
    display_predict.plot(cmap='Reds')
    plt.title(f'Predict-then-Explain Confusion Matrix\nAccuracy: {accuracy_predict:.3f}')
    plt.savefig('results/predict_then_explain_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    explain_conf_matrix()
    predict_conf_matrix()
