import numpy as np


def k_fold(dataset,k):
    folds=np.array_split(np.array(dataset,dtype=object),k)
    result=[]
    for i in range(k):
        tr=np.concatenate((*folds[:i],*folds[i+1:]),dtype=object)
        vl=folds[i]
        result.append((tr,vl))
    return result


def evaluate_performance(model,dataset,loss_function,classification_function):
    outputs=[]
    targets=[]
    for pattern in dataset.get_dataset():
        input_val=pattern[0]
        target_val=pattern[1]
        res=model.predict(input_val)
        targets.append(target_val)
        outputs.append(res)
    return loss_function(np.array(outputs),np.array(targets)),classification_function(np.array(outputs),np.array(targets))

def monk_classification(o,t):
    precision=0
    for target,value in zip(t,o):
        classification= np.array([1] if value[0] >= 0.5 else [0] ) 
        if np.array_equal(classification,target):
            precision+=1
    return precision/len(t)