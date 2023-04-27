## car table

import urllib.request as urllib2
from bs4 import BeautifulSoup
import csv
import numpy as np
import pandas as pd
import plotly.express as px


def ranker(scores, from_bottom = True):
    #get index 
    rank = np.zeros(scores.shape, dtype = int)
    arg_sort = np.argsort(scores)
    if from_bottom == False:
        arg_sort = arg_sort[::-1]
    for i in range(0,len(scores)):
        rank[i] = int(np.where(arg_sort==i)[0])
    return(rank)
    
#FUZZY topsis
def fuzzytopsis(data, objective_data, weight_criteria):
    import timeit
    import numpy as np
    start = timeit.default_timer()

    #normalise weights
    if len(weight_criteria.shape)>=2: 
        len_w = len(weight_criteria[0])
    else:
        len_w = len(weight_criteria)
    
    for i in range(0,len_w):
        weight_criteria[:,i] = (weight_criteria[:,i])/sum(weight_criteria[:,i])
    
    #create empty dataframe
    data_WN = np.zeros((data.shape[0],data.shape[1],weight_criteria.shape[1]))
    #normalize depending on objective of criteria, and weigh the data 
    for i in range(0, data.shape[0]):
        R_i = (data[i,:]/np.sum(np.square(data[i,:]))**0.5)
        data_WN[i,:,:] = np.matrix(R_i).transpose() * np.matrix(weight_criteria[i,:])
    #find ideal best and worst solutions (max and min depending on objective for 
    #each criteira). Assign these in new df. note, 2 is one for both best and worst 
    ideal_criteria = np.zeros((data.shape[0], 2, weight_criteria.shape[1]))
    for j in range(0, data.shape[0]):
        if objective_data[j] == 1:
            for i in range(0,weight_criteria.shape[1]):
                ideal_criteria[j,0,i] = max(data_WN[j,:,i])
                ideal_criteria[j,1,i] = min(data_WN[j,:,i])
        else:
            for i in range(0,weight_criteria.shape[1]):
                ideal_criteria[j,0,i] = min(data_WN[j,:,i])
                ideal_criteria[j,1,i] = max(data_WN[j,:,i])
    
    #compute seperation measure
    seperation_matrix = np.zeros((2,data.shape[1], weight_criteria.shape[1]))
    for k in range(0, data.shape[1]):
        for i in range(0,weight_criteria.shape[1]):
            seperation_matrix[:,k,i] = np.sqrt(np.sum(np.square(np.array([data_WN[:,k,i],data_WN[:,k,i]]).transpose()-ideal_criteria[:,:,i]), axis = 0))
    
    #measure relative closeness i.e. worst/(worst+best)
    relative_closeness = np.zeros((data.shape[1],weight_criteria.shape[1]))
    for l in range(0, data.shape[1]):
        for i in range(0,weight_criteria.shape[1]):
            relative_closeness[l,i] = seperation_matrix[1,l,i]/np.sum(seperation_matrix[:,l,i])
    #score is basically the length to the worst, so the higher value the better!
    #geometric mean of all the closeness coeficients for each alternative
    final_closeness = np.zeros(data.shape[1])
    for i in range(0,data.shape[1]):
        final_closeness[i] = relative_closeness[i,:].prod()**(1/len(relative_closeness[i,:]))
    
    rank = ranker(final_closeness, from_bottom = False)
    end = timeit.default_timer()
    time_topsis = end - start
    
    fuzzytopsis.rank = rank
    fuzzytopsis.cci = relative_closeness
    fuzzytopsis.ccfinal = final_closeness
    fuzzytopsis.time = time_topsis
    return(fuzzytopsis)




def AHP(data, objective_data, weight_criteria = None):
    import timeit
    import numpy as np
    start = timeit.default_timer()
    data_overall = np.zeros((data.shape[1],data.shape[0]))
    data_temp = np.zeros((data.shape[1],data.shape[1]))
    data_orient = np.zeros((data.shape))
    #objective data orientation
    for i in range(0,data.shape[0]):
        if objective_data[i]==0: #min
            data_orient[i,:] = 1.25*np.max(data[i,:])-data[i,:]
        if objective_data[i]==1: #max
            data_orient[i,:] = data[i,:]
                        
    #score for each alternative
    for i in range(0,data.shape[0]):
        #find score 
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[1]):
                data_temp[j,k] = data_orient[i,j] / data_orient[i,k]
        #normalize
        for l in range(0, data.shape[1]):
            data_temp[:, l] = data_temp[:, l]/sum(data_temp[:, l])
        #average 
        for m in range(0, data.shape[1]):
            data_overall[m,i] = data_temp[m,:].mean() 
    
    #overall score computed with the weight for each criteria set by the dm 
    if any(weight_criteria == None):
        weight_criteria = [1]*data.shape[0]
        weight_criteria = np.asarray(weight_criteria)/sum(weight_criteria)
        
    ODscore = np.matmul(data_overall, weight_criteria)
    ODrank = ranker(ODscore, from_bottom = False)
    end = timeit.default_timer()
    time_AHP = end - start
    #note the last is the biggest (and begins from 0)
    AHP.rank = ODrank
    AHP.time = time_AHP
    return(AHP)




## TOPSIS ##

#topsis contains also multiple objectives, i.e. both maximization and 
#minimization of some criteria. A vector representing each criterias objective
#is therefore created.

#objective_data = np.random.randint(low = 0, high = 2, size = data.shape[0])
#objective_data = np.array([1,1,1,1,1])
#1 indicates a max objective, and 0 a minimisation objective

#topsis normalises data (also negative elements), weights the normalised data, 
#and creates a seperation measure from optimal worst and best solutions, and 
#then ranks the measures with regard to their relative closeness.

def topsis(data, objective_data, weight_criteria):
    import timeit
    import numpy as np
    start = timeit.default_timer()

    #assign weights and normalize them - should be done by DM. Here equal
    if any(weight_criteria == None):
        weight_criteria = [1]*data.shape[0]
        weight_criteria = np.asarray(weight_criteria)/sum(weight_criteria)

    #create empty dataframe
    data_WN = np.zeros((data.shape[0],data.shape[1]))
    #normalize and weigh the data 
    for i in range(0, data.shape[0]):
        data_WN[i,:] = (data[i,:]/np.sum(np.square(data[i,:]))**0.5) * weight_criteria[i]
    #find ideal best and worst solutions (max and min depending on objective for 
    #each criteira). Assign these in new df. note, 2 is one for both best and worst 
    ideal_criteria = np.zeros((data.shape[0], 2))
    for j in range(0, data.shape[0]):
        if objective_data[j] == 1:
            ideal_criteria[j,0] = max(data_WN[j,:])
            ideal_criteria[j,1] = min(data_WN[j,:])
        else:
            ideal_criteria[j,0] = min(data_WN[j,:])
            ideal_criteria[j,1] = max(data_WN[j,:])
    #compute seperation measure
    seperation_matrix = np.zeros((2,data.shape[1]))
    for k in range(0, data.shape[1]):
        seperation_matrix[:,k] = np.sqrt(np.sum(np.square(np.array([data_WN[:,k],data_WN[:,k]]).transpose()-ideal_criteria), axis = 0))
    
    #measure relative closeness i.e. worst/(worst+best)
    relative_closeness = np.zeros(data.shape[1])
    for l in range(0, data.shape[1]):
        relative_closeness[l] = seperation_matrix[1,l]/np.sum(seperation_matrix[:,l])
    #score is basically the length to the worst, so the higher value the better!
    rank = ranker(relative_closeness, from_bottom = False)
    end = timeit.default_timer()
    time_topsis = end - start
    
    topsis.score = relative_closeness
    topsis.rank = rank
    topsis.time = time_topsis
    return(topsis)
    
    


## ELECTRE ##
#will propably let way to many alternatives be better than the others..

#electre normalizes and weighs the data. It then deals with the issue of 
#outranking in the regard of concordance and discordance set, i.e. it tries
#to find incentive that ranks alternatives with one over another.

def parallelectre(dat, q, p, v, objective_data, weights):
    import numpy as np
    import timeit
    start = timeit.default_timer()
        
    #normalise data, create empty dataframe
    data_WN = np.copy(dat) #np.zeros((data.shape[0],data.shape[1]))
    data_shape = data_WN.shape
    
    #objectify data
    for i in range(0,data_shape[0]):
        if objective_data[i] == 0:
            data_WN[i,:] = np.max(data_WN[i,:])-data_WN[i,:]    
        
    #calculate concordance 
    CM_temp = np.zeros((data_shape[1],data_shape[1]))
    S_temp = np.zeros((data_shape[1],data_shape[1]))
    qq = np.array(data_shape[1]*list(q)).reshape(data_shape[1], data_shape[0]).T
    pp = np.array(data_shape[1]*list(p)).reshape(data_shape[1], data_shape[0]).T
    vv = np.array(data_shape[1]*list(v)).reshape(data_shape[1], data_shape[0]).T
    
    for i in range(0, data_shape[1]):                     #i is a, j is all other
        #for j in range(0, data.shape[1]):
        #if i==j:
        #CM_temp[i,i] = 1
        #S_temp[i,i] = 1
        #else:
        diff_vector = data_WN - np.array(data_shape[1]*list(data_WN[:,i])).reshape(data_shape[1], data_shape[0]).T
        #concordance if
        phi = np.zeros(diff_vector.shape)
        phi[diff_vector <= qq] = 1 
        a = (qq < diff_vector) & (diff_vector < pp)
        phi[a] = (pp[a] - diff_vector[a])/(pp[a]-qq[a])
        #phi[pp <= diff_vector] = 0          #these are already denoted 0
        #discordance if
        d = np.zeros(diff_vector.shape)
        d[diff_vector >= vv] = 1
        b = (pp < diff_vector) & (diff_vector < vv)
        d[b] = (diff_vector[b]-pp[b])/(vv[b]-pp[b])
        #d[diff_vector<=p] = 0              #these are already denoted 0
        
        #overall concordance and credibility index
        CM_temp[i,:] = np.dot(weights, phi)/np.sum(weights)
        CC = np.array(data_shape[0]*list(CM_temp[i,:])).reshape(data_shape[0], data_shape[1])
        
        all_true = np.sum(d<=CC, axis = 0) == data_shape[1]
        S_temp[i, all_true] = np.array(CM_temp[i,all_true])
        
        K = ((d>CC) & (CC != 1))[:,~all_true]
        prod_m = ~K*1
        #upper
        prod_m_upper = np.array(prod_m, dtype = "float64")
        #np.put(prod_m_upper, np.squeeze(np.where(K.reshape((K.shape[0]*K.shape[1]))))[np.squeeze(np.where(~all_true))], (1-d[K])[~all_true])
        np.put(prod_m_upper, [np.where(K.reshape((K.shape[0]*K.shape[1])))], (1-d[:,~all_true][K]))
        upper = np.prod(prod_m_upper.reshape(K.shape), axis = 0)
        #lower
        #prod_m_lower = np.array(prod_m, dtype = "float64")
        #np.put(prod_m_lower, [np.where(K.reshape((K.shape[0]*K.shape[1])))], (1-CM_temp[i,~all_true]))
        #lower = np.prod(prod_m_lower.reshape(K.shape), axis = 0)
        
        lower = (1-CM_temp[i,~all_true])**(np.sum(K, axis =0))
                  
        S_temp[i,~all_true] = CM_temp[i,~all_true] * (upper/lower)
        
    #print(np.array_str(CM_temp, precision = 2, suppress_small=True))
    #print(np.array_str(S_temp,  precision = 2, suppress_small=True))
    score = np.mean(S_temp, axis = 1)

    
    end = timeit.default_timer()
    time_electre = end - start
    
    #electreiii.description = "Of a to b, 3 means prefered, 2 means indifferent, 1 means incompatible, and 0 means not prefered over. Remember that for prelimenary rankings that Python counts from 0"
    parallelectre.time = time_electre
    parallelectre.rank = ranker(score, from_bottom = False)
    
    #electreiii.top = D_distil_rank
    #electreiii.bottom = A_distil_rank
    #electreiii.final = final_ranking
    return(parallelectre)
    
    
#WSA

def wsa(data, objective_data, weight_criteria):
    import timeit
    import numpy as np
    start = timeit.default_timer()

    #create empty dataframe
    data_WN = np.zeros((data.shape[0],data.shape[1]))
    
    #normalize depending on objective of criteria, and weigh the data 
    for i in range(0, data.shape[0]):
        if all(data[i,:] == 0):
            data_WN[i,:] = np.zeros((data.shape[1]))
            continue
        elif objective_data[i] == 1:
            R_i = (data[i,:]/np.sum(np.square(data[i,:]))**0.5)
        else:
            R_i = (1/data[i,:]/np.sum(np.square(1/data[i,:]))**0.5)
        data_WN[i,:] = R_i * weight_criteria[i]
    
    
    #sum the data
    wsa_score = np.sum(data_WN, axis = 0)
    rank = ranker(wsa_score, from_bottom = False)
    end = timeit.default_timer()
    time_wsa = end - start
    
    wsa.rank = rank
    wsa.score = wsa_score
    wsa.time = time_wsa
    return(wsa)


#data scrape

REQUEST_HEADER = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                  (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36"}

#url = "https://www.bilbasen.dk/find-en-forhandler/bilforhandler-kurt-thomsen-automobiler-id3115"
url = "https://www.bilbasen.dk/brugt/bil?IncludeEngrosCVR=true&PriceFrom=0&includeLeasing=false&free=tesla&Cartypes=Sedan&IncludeCallForPrice=false"

req = urllib2.Request(url, headers=REQUEST_HEADER)
page = urllib2.urlopen(req, timeout=20).read()

soup = BeautifulSoup(page, "html.parser")

name = soup.find_all("a", {"class": "listing-heading darkLink"})
name_list = list()
for i in range(len(name)):
    s1 = str(name[i]).find('href="')
    s2 = str(name[i]).find('">')
    link = str(name[i])[s1+6:s2]
    name_list.append(link)

kmt_år = soup.find_all("div", {"class": "col-xs-2 listing-data"})
len(kmt_år)
dat = np.zeros((3,int(len(kmt_år)/3)))
idx=0
for i in range(len(kmt_år)):
    if i%3 == 0:
        continue
    if i%3 == 1: #kmt
        str_kmt = (str(kmt_år[i])[35:][:-6]).replace('.','')
        dat[0,idx] = int(str_kmt)
    if i%3 == 2: #år
        str_år = str(kmt_år[i])[35:][:-6]
        dat[1,idx] = int(str_år)
        idx += 1


pris = soup.find_all("div", {"class": "col-xs-3 listing-price"})
for i in range(len(pris)):
    str_kmt = (str(pris[i])[36:][:-9]).replace('.','')
    dat[2,i] = int(str_kmt)








##################### score ####################################
obj_dat = [0,1,0]

#fuzzy topsis
w = np.array([[0.6,0.3,0.1],
              [0.4,0.3,0.3],
              [0.4,0.1,0.5]])
score1 = fuzzytopsis(data = dat, objective_data = obj_dat, weight_criteria = w)
score1.rank

#ftopsis2 
w = np.array([[0.4,0.3,0.3],
              [0.4,0.3,0.3],
              [0.4,0.3,0.3]])
score2 = fuzzytopsis(data = dat, objective_data = obj_dat, weight_criteria = w)
score2.rank

#topsis
w = np.array([0.4,0.3,0.3])
score3 = topsis(data = dat, objective_data = obj_dat, weight_criteria = w)
score3.rank

#AHP simplified
w = np.array([0.4,0.3,0.3])
score4 = AHP(data=dat, objective_data = obj_dat, weight_criteria = w)
score4.rank 

#electre parallel
w = np.array([0.4,0.3,0.3])
q = [0,0,0]
p = [0,0,0]
v = [100000, 100000, 10000]
score5 = parallelectre(dat=dat, q=q, p=p, v=v, objective_data = obj_dat, weights = w)
score5.rank 

#WSA
w = np.array([0.4,0.3,0.3])
score6 = wsa(data = dat, objective_data = obj_dat, weight_criteria = w)
score6.rank

w = np.array([0.6,0.3,0.1])
score7 = wsa(data = dat, objective_data = obj_dat, weight_criteria = w)
score7.rank

w = np.array([0.4,0.1,0.5])
score8 = wsa(data = dat, objective_data = obj_dat, weight_criteria = w)
score8.rank


df = pd.DataFrame({
        'ftopsis':  score1.rank, 
        'ftopsis2': score2.rank, 
        'topsis':   score3.rank, 
        'ahp':      score4.rank,
        'electre':  score5.rank,
        'wsa1':     score6.rank, 
        'wsa2':     score7.rank, 
        'wsa3':     score8.rank
})


#################### PLOT ######################################

from plotly.offline import plot

df_plot = pd.DataFrame({
        'name': name_list, 
        'pris': dat[2,:], 
        'km':   dat[0,:], 
        'år':   dat[1,:],
        'rank':score4.rank   #make sure to change index of score depending on which mcdm method you want!
})
df.head()


fig = px.scatter_3d(df_plot, x='pris', y='år', z='km', color = 'rank',
                    hover_name = 'name',
                    title="cars")
plot(fig)

