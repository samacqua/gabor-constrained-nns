import os
import numpy as np
import scipy.stats

directory = "test_experiment_responses"

cifar_key = ['5', '4', '6', '7', '5', '8', '9', '6', '5', '5']
fashion_mnist_key = ['4', '6', '9', '3', '1', '2', '3', '0', '5', '1']

correct_proportions = []

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def same(answers, answer_key):
    correct = []
    for answer, key in zip(answers, answer_key):
        if answer == key:
            correct.append(1)
        else:
            correct.append(0)
    return np.array(correct)

for filename in os.listdir(directory):
    if filename[0] != '.':
        with open(directory+"/"+filename, "r") as f:
            raw_data = f.read()
        extracted_data = raw_data.split(',')

    cifar_cnn_answers = extracted_data[:10]
    cifar_gabor_answers = extracted_data[10:20]
    fashion_cnn_answers = extracted_data[20:30]
    fashion_gabor_answers = extracted_data[30:]
    assert len(cifar_cnn_answers) == 10
    assert len(cifar_gabor_answers) == 10
    assert len(fashion_cnn_answers) == 10
    assert len(fashion_gabor_answers) == 10
    

    cifar_cnn_proportion = np.mean(same(cifar_cnn_answers, cifar_key))
    cifar_gabor_proportion = np.mean(same(cifar_gabor_answers, cifar_key))
    fashion_cnn_proportion = np.mean(same(fashion_cnn_answers, fashion_mnist_key))
    fashion_gabor_proportion = np.mean(same(fashion_gabor_answers, fashion_mnist_key))


    correct_proportions.append({'cifar_cnn': cifar_cnn_proportion,
                                'cifar_gabor': cifar_gabor_proportion,
                                'fashion_cnn': fashion_cnn_proportion,
                                'fashion_gabor': fashion_gabor_proportion})


cifar_cnn_population = np.array([d['cifar_cnn'] for d in correct_proportions])
fashion_cnn_population = np.array([d['fashion_cnn'] for d in correct_proportions])

cifar_gabor_population = np.array([d['cifar_gabor'] for d in correct_proportions])
fashion_gabor_population = np.array([d['fashion_gabor'] for d in correct_proportions])

total_cnn_population = (cifar_cnn_population + fashion_cnn_population)/2
assert len(total_cnn_population) == len(os.listdir(directory))

total_gabor_population = (cifar_gabor_population + fashion_gabor_population)/2
assert len(total_gabor_population) == len(os.listdir(directory))

print(f'------RESULTS:------')

mean, lower, upper = mean_confidence_interval(cifar_cnn_population)
print(f'\nmean cifar adversary performance based on cnn:{np.mean(mean)} +/- {upper-mean}')
mean, lower, upper = mean_confidence_interval(fashion_cnn_population)
print(f'\nmean fashion mnist adversary performance based on cnn:{np.mean(mean)} +/- {upper-mean}')

mean, lower, upper = mean_confidence_interval(cifar_gabor_population)
print(f'\nmean cifar adversary performance based on Gabornet:{np.mean(mean)} +/- {upper-mean}')
mean, lower, upper = mean_confidence_interval(fashion_gabor_population)
print(f'\nmean fashion mnist adversary performance based on Gabornet:{np.mean(mean)} +/- {upper-mean}')

# student t test:
all_out = scipy.stats.ttest_ind(total_cnn_population, total_gabor_population, equal_var=True)
print(f'\nrunning a student t-test for cnn vs gabor, we get a p-value of {all_out.pvalue} and a statistic of {all_out.statistic}')

cifar_only_out = scipy.stats.ttest_ind(cifar_cnn_population, cifar_gabor_population, equal_var=True)
print(f'\nusing only cifar, we get a p-value of {cifar_only_out.pvalue} and a statistic of {cifar_only_out.statistic}')

fashion_only_out = scipy.stats.ttest_ind(fashion_cnn_population, fashion_gabor_population, equal_var=True)
print(f'\nusing only fashion, we get a p-value of {fashion_only_out.pvalue} and a statistic of {fashion_only_out.statistic}')
