import timeit
import numpy as np
import time
from sklearn import svm
from tqdm import tqdm


NB_class=4


def Load():
    data = np.loadtxt("defautsrails.dat")
    X = data[:,:-1]  # tout sauf la dernière colonne
    y = data[:,-1]  # uniquement la dernière colonne
    return X,y

def Reetiquetage(Y,Class):
    Y1=np.ones(len(Y))
    Y1[Y!=Class]=-1
    return Y1

def L_model(Xapp,Yapp,Class,C):
    Yappr=Reetiquetage(Yapp,Class)
    modelsvm = svm.LinearSVC(C=C,max_iter=10000000)
    modelsvm.fit(Xapp,Yappr)
    return modelsvm

def Models(X,Y,C):
    model=[L_model(X,Y,i+1,C[i]) for i in range(0,NB_class)]
    return model

def Models_pred(Xtest,models):
    Ypreds=[m.predict(Xtest) for m in models]
    return Ypreds

def Targets(Y):
    return [Reetiquetage(Y,i+1) for i in range(0,NB_class)]

def Err(X,Y,models):
    Pred=Models_pred(X,models)
    T=Targets(Y)
    E=[Pred[i]!=T[i] for i in range(0, NB_class)]
    return np.mean(E,axis=1)


class un_contre_tous:

    def apprentisage(self,X,Y,C):
        self.models=Models(X,Y,C)

    def predection(self,X):
        U=[model.decision_function(X) for model in self.models]
        return np.argmax(U,axis=0)+1

    def err(self,X,Y):
        return np.mean(self.predection(X)!=Y)

def LOO(X,Y,C,progress_bar=False):
  model = un_contre_tous()
  err_multiclass=0
  err_bin=np.zeros(NB_class)
  if(progress_bar):
      pbar=tqdm(total=len(Y),desc="LOO : ")
  for i in range(len(Y)):
      Xn,Yn=np.delete(X,i,axis=0),np.delete(Y,i)
      model.apprentisage(Xn,Yn,C)
      err_multiclass+=model.predection([X[i]])[0]!=Y[i]
      for j in range(NB_class):
          p=model.models[j].predict([X[j]])[0]
          err_bin[j]+=((Y[i]!=j+1 and p==j)or(Y[i]==j+1 and not p==j))
          #print("LOO it:{} nb_err:{} nb_err-bin:{} ".format(i, err_multiclass, err_bin))
      if (progress_bar):
        pbar.update(1)
  if (progress_bar):
    pbar.close()
  return err_multiclass/len(Y),err_bin/len(Y)

def LOO_bin(X,Y,C,k):
    err = 0
    for i in range(len(Y)):
        Xn, Yn = np.delete(X, i, axis=0), np.delete(Y, i)
        model=L_model(Xn,Yn,k,C)
        pred=model.predict([X[i]])[0]
        err += not(((pred==1)and(Y[i]==k))or((pred!=1)and(Y[i]!=k)))
    #print(C,k,err/len(Y))
    return err/len(Y)

def select_C_ind(X,Y,C_values,progress_bar=False):
    best_err=[np.math.inf]*NB_class
    best_c=[0]*NB_class
    if(progress_bar):
        pbar = tqdm(total=len(C_values)*NB_class, desc="Sélection de modèle indépendament : ")
    for i in range(NB_class):
        for c in C_values:
            err=LOO_bin(X,Y,c[i],i+1)
            if(err<best_err[i]):
                best_c[i],best_err[i]=c[i],err
            if (progress_bar):
                pbar.update(1)
    if (progress_bar):
        pbar.close()
    return best_c,best_err

def select_C(X,Y,C_values,progress_bar=False):
    best_err=np.math.inf
    best_c=None
    if (progress_bar):
        pbar = tqdm(total=len(C_values), desc="Sélection de modèle : ")
    for c in C_values:
        err=LOO(X,Y,c)[0]
        #print("///////////",c,"/////////",err)
        if(err<best_err):
            best_c,best_err=c,err
        if (progress_bar):
            pbar.update(1)
    if (progress_bar):
        pbar.close()
    return best_c,best_err

def Best_model(X,Y,C_values,Ind=False,progress_bar=False):
    err=0
    model = un_contre_tous()
    if (progress_bar):
        pbar = tqdm(total=len(Y), desc="Validation modéle (& Sélection de modèle) : ")
    for i in range(len(Y)):
        Xn,Yn=np.delete(X,i,axis=0),np.delete(Y,i)
        if(not Ind):
            bestc,e=select_C(Xn,Yn,C_values)
        else:
            bestc,e=select_C_ind(Xn,Yn,C_values)
        model.apprentisage(Xn,Yn,bestc)
        err+=model.predection([X[i]])!=Y[i]
        if (progress_bar):
            pbar.update(1)
        #print("\n \n bestc {} ".format(bestc))
    if (progress_bar):
        pbar.close()
    if (not Ind):
        bestc, e = select_C(X, Y, C_values)
    else:
        bestc, e = select_C_ind(X, Y, C_values)
    model.apprentisage(X,Y,bestc)

    return model,err/len(Y),bestc



def main():
    X, Y = Load()
    '''
    #########################
    print("////////////////////// Evaluation des modéles binaire sur la base d'apprentissage ////////// \n")
    for i in range(4):
        C=0.1*(10**i)
        models=Models(X,Y,[C,C,C,C])
        err=Err(X,Y,models)
        print(" C={}  Err_bin_1={}% Err_bin_2={}% Err_bin_3={}% Err_bin_4={}%".format(C,err[0]*100,err[1]*100,err[2]*100,err[3]*100))
    print("\n////////////////////////////////////////////////////////////////////// \n")
    ##########################

    #########################
    print("////////////////////// Evaluation des modéles un_contre_tous ////////// \n")
    for i in range(3):
        C=0.1*(10**i)
        t=time.time()
        model=un_contre_tous()
        model.apprentisage(X,Y,[C,C,C,C])
        err_app=model.err(X,Y)
        err_val=LOO(X,Y,[C,C,C,C],progress_bar=True)
        t=time.time()-t
        print("///////// C={}  \n Err_Appresntisagge = {}% \n Err_Validation_global(LOO) = {}%".format(C,err_app,err_val[0]*100))
        print(" Err_Validation(LOO)_classifieur binaire : \n  Err_k1={}% Err_k2={}% Err_k3={}% Err_k4={}%".format(err_val[1][0]*100,err_val[1][1]*100,err_val[1][2]*100,err_val[1][3]*100))
        print("///// Temps de calcule ={} //////".format(t))
    print("\n////////////////////////////////////////////////////////////////////// \n")
    ###########################


    ###########################

    print("//////////////////////  Sélection de modèle le meme C pour les 4 modéle ////////// \n")
    t = time.time()
    c_values = [[0.1*(10**i),0.1*(10**i),0.1*(10**i),0.1*(10**i)] for i in range(4)]
    bestc, err = select_C(X, Y, c_values, progress_bar=True)
    t = time.time() - t
    print("///// Grilles Valeurs pour C //////", c_values)
    print("Meilleurs C =( {} , {} , {} , {} ) , avec erreurs ={}%".format(bestc[0], bestc[1], bestc[2], bestc[3], err*100))
    print("///// Temps de calcule = {}//////\n".format(t))


    ###########################

    ###########################

    print("//////////////////////  Sélection de modèle indépendamment ////////// \n")
    t=time.time()
    c_values = [[0.1*(10**i),0.1*(10**i),0.1*(10**i),0.1*(10**i)] for i in range(4)]
    bestc,err=select_C_ind(X , Y , c_values , progress_bar=True)
    t= time.time()-t
    print("///// Grilles Valeurs pour C //////",c_values)
    print("Meilleurs C =( {} , {} , {} , {} ) , avec erreurs =( {}% , {}% , {}% , {}% )".format(bestc[0],bestc[1],bestc[2],bestc[3],err[0]*100,err[1]*100,err[2]*100,err[3]*100))
    print("///// Temps de calcule ={} //////\n".format(t))

    ###########################

    
    ###########################

    print("//////////////////////  Sélection de modèle simultanément ////////// \n")
    t = time.time()
    c_values =[[0.1*(10**c1),0.1*(10**c2),0.1*(10**c3),0.1*(10**c4)] for c1 in range(4) for c2 in range(4) for c3 in range(4) for c4 in range(4)]
    bestc, err = select_C(X, Y, c_values, progress_bar=True)
    t = time.time() - t
    print("///// Grilles Valeurs pour C //////",np.array(c_values))
    print("Meilleurs C =( {} , {} , {} , {} ) , avec erreurs ={}%".format(bestc[0], bestc[1], bestc[2],bestc[3], err*100))
    print("///// Temps de calcule = {}//////\n".format(t))

    ###########################
    '''
    ###########################

    print("////////////////////// Validation & Sélection de modèle indépendamment ////////// \n")
    t = time.time()
    c_values=[[i+1,i+1,i+1,i+1] for i in range(2)]
    model,err,bestc = Best_model(X,Y,c_values,Ind=True,progress_bar=True)
    t = time.time() - t
    print("///// Grilles Valeurs pour C //////",np.array(c_values))
    print("Meilleurs C =( {} , {} , {} , {} ) , avec erreurs ={}%".format(bestc[0], bestc[1], bestc[2], bestc[3], err*100))
    print("///// Temps de calcule = {}//////\n".format(t))

    ###########################

    ###########################


    print("////////////////////// Validation &  Sélection de modèle simultanément ////////// \n")
    t = time.time()
    c_values = [[c1+1,c2+1,c3+1,c4+1]for c1 in range(2) for c2 in range(2) for c3 in range(2) for c4 in
                range(2)]
    model,err,bestc=Best_model(X,Y,c_values,progress_bar=True)
    t = time.time() - t
    print("///// Grilles Valeurs pour C //////",np.array(c_values))
    print("Meilleurs C =( {} , {} , {} , {} ) , avec erreurs ={}%".format(bestc[0], bestc[1], bestc[2], bestc[3], err*100))
    print("///// Temps de calcule ={} //////\n".format(t))

    ###########################



main()