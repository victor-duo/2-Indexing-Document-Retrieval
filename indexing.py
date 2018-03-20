# import
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# prepare corpus
corpus = []
for d in range(1400):
    f = open("./d/"+str(d+1)+".txt")
    corpus.append(f.read())
# add query to corpus
for q in range(1,226):
    f = open("./q/"+str(q)+".txt")
    corpus.append(f.read())
# init vectorizer
#binary representation
bool_vectorizer = TfidfVectorizer(binary=True,norm=None,use_idf=False)
#pure term frequency
tf_vectorizer = TfidfVectorizer(use_idf=False)
#TF-IDF
tfidf_vectorizer = TfidfVectorizer()
# prepare matrix
bool_matrix = bool_vectorizer.fit_transform(corpus)
tf_matrix = tf_vectorizer.fit_transform(corpus)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)


#measures
topRelevantEucl=[]
topRelevantCos=[]
euclBool=[]
cosBool=[]
euclTf=[]
cosTf=[]
euclTfIdf=[]
cosTfIdf=[]
for i in range(0,225):
    #Euclidean distances
    topRelevantEuclQuery=[]
    simEuclbool=np.array(euclidean_distances(bool_matrix[len(corpus)-225+i],bool_matrix[0:len(corpus)-225])[0])
    simEucltf=np.array(euclidean_distances(tf_matrix[len(corpus)-225+i],tf_matrix[0:len(corpus)-225])[0])
    simEucltfidf=np.array(euclidean_distances(tfidf_matrix[len(corpus)-225+i],tfidf_matrix[0:len(corpus)-225])[0])
    euclBool.append(simEuclbool)
    euclTf.append(simEucltf)
    euclTfIdf.append(simEucltfidf)
    topRelevantEuclQuery.append(simEuclbool.argsort()[-10:][::-1]+1)
    topRelevantEuclQuery.append(simEucltf.argsort()[-10:][::-1]+1)
    topRelevantEuclQuery.append(simEucltfidf.argsort()[-10:][::-1]+1)
    topRelevantEucl.append(topRelevantEuclQuery)
    #Cosine similarity
    topRelevantCosQuery=[]
    simCosbool = np.array(cosine_similarity(bool_matrix[len(corpus)-225+i], bool_matrix[0:(len(corpus)-225)])[0])
    simCostf = np.array(cosine_similarity(tf_matrix[len(corpus)-225+i], tf_matrix[0:(len(corpus)-225)])[0])
    simCostfidf = np.array(cosine_similarity(tfidf_matrix[len(corpus)-225+i], tfidf_matrix[0:(len(corpus)-225)])[0])
    cosBool.append(simCosbool)
    cosTf.append(simCostf)
    cosTfIdf.append(simCostfidf)
    topRelevantCosQuery.append(simCosbool.argsort()[-10:][::-1]+1)
    topRelevantCosQuery.append(simCostf.argsort()[-10:][::-1]+1)
    topRelevantCosQuery.append(simCostfidf.argsort()[-10:][::-1]+1)
    topRelevantCos.append(topRelevantCosQuery)


euclBool=np.array(euclBool).T
euclTf=np.array(euclTf).T
euclTfIdf=np.array(euclTfIdf).T
cosBool=np.array(cosBool).T
cosTf=np.array(cosTf).T
cosTfIdf=np.array(cosTfIdf).T

queryHeader="Query Number : "
for i in range(1,225):
    queryHeader+=str(i)+","
queryHeader+=str(225)

#Saving in csv format
np.savetxt("outputtfidfcosine.csv",cosTfIdf,delimiter=",",header=queryHeader,fmt='%10.5f')
np.savetxt("outputtfcosine.csv",cosTf,delimiter=",",header=queryHeader,fmt='%10.5f')
np.savetxt("outputboolcosine.csv",cosBool,delimiter=",",header=queryHeader,fmt='%10.5f')
np.savetxt("ouputbooleuclidean.csv",euclBool,delimiter=",",header=queryHeader,fmt='%10.5f')
np.savetxt("outputtfeuclidean.csv",euclTf,delimiter=",",header=queryHeader,fmt='%10.5f')
np.savetxt("outputtfidfeuclidean.csv", euclTfIdf, delimiter=",",header=queryHeader,fmt='%10.5f') #we set a width of 10 and precision of 5


#Compute precision, recall & F-measure
#We do it with the top-10 documents
relevantDocumentID=[]
for d in range(225):
    f = open("./r/"+str(d+1)+".txt")
    relevantDocumentID.append(f.read())
for i in range(225):
    relevantDocumentID[i]=relevantDocumentID[i].replace('\n',',')
    relevantDocumentID[i]=relevantDocumentID[i][:len(relevantDocumentID[i])-1]
counterID=[]
precision=[]
recall=[]

for i in range(0,225):
    nbRelevantDocument=len(relevantDocumentID[i])
    precisionQuery=[]
    recallQuery=[]
    for j in range(0,3):
        # we compute for the binary, the tf and the tfidf
        counterForOneQuery=0
        p=0
        r=0
        for id in topRelevantEucl[i][j]:
            if str(id) in relevantDocumentID[i]:
                counterForOneQuery+=1
        counterID.append(counterForOneQuery)
        p=(counterForOneQuery/(counterForOneQuery+(10-counterForOneQuery)))
        r=(counterForOneQuery/(counterForOneQuery+(nbRelevantDocument-counterForOneQuery)))
        precisionQuery.append(p)
        recallQuery.append(r)
        p=0
        r=0
        counter2=0
        for id in topRelevantCos[i][j]:
            if str(id) in relevantDocumentID[i]:
                counter2+=1
        counterID.append(counter2)
        p=(counter2/(counter2+(10-counter2)))
        r=(counter2/(counter2+(nbRelevantDocument-counter2)))
        precisionQuery.append(p)
        recallQuery.append(r)
    precision.append(precisionQuery)
    recall.append(recallQuery)
#precision,recall,FMeasure are given in this order
#binaryEucl,binaryCos,tfEucl,tfCos,tfidfEucl,tfidfCos
#print(counterID, precision, recall)
FMeasure=[]
for i in range(0,len(precision)):
    FMeasureQuery=[]
    for j in range(0,len(precision[i])):
        if(precision[i][j]+recall[i][j]!=0):
            FMeasureQuery.append(2*(precision[i][j]*recall[i][j])/(precision[i][j]+recall[i][j]))
        else:
            FMeasureQuery.append(0)
    FMeasure.append(FMeasureQuery)
#print(FMeasure)
precision=np.array(precision)
recall=np.array(recall)
FMeasure=np.array(FMeasure)
np.savetxt("precision.csv",precision,delimiter=",",fmt='%10.5f')
np.savetxt("recall.csv",recall,delimiter=",",fmt='%10.5f')
np.savetxt("FMeasure.csv",FMeasure,delimiter=",",fmt='%10.5f')
avgPrecision=[]
avgRecall=[]
avgFMeasure=[]
for i in range(0,len(precision[0])):
    sumP=0
    sumR=0
    sumF=0
    for j in range(0,len(precision)):
        sumP+=precision[j][i]
        sumR+=recall[j][i]
        sumF+=FMeasure[j][i]
    sumP/=225
    sumR/=225
    sumF/=225
    avgPrecision.append(sumP)
    avgRecall.append(sumR)
    avgFMeasure.append(sumF)
print(avgPrecision)
print(avgRecall)
print(avgFMeasure)
