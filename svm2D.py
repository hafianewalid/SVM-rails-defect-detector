import numpy as np
import matplotlib.pyplot as plt
import math
import os

from sklearn import svm



#### programme principal à implémenter dans cette fonction ####
def monprogramme(Xapp, Yapp, C):
    """
		Programme pour les SVM linéaires (lancé avec ESPACE)

		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
	"""
    print("Apprentissage lancé avec " + str(len(Xapp)) + " points et C = ", C)

    # à compléter pour apprendre le modèle SVM...

    modelsvm = svm.LinearSVC(C=C, max_iter=10000)
    modelsvm.fit(Xapp, Yapp)

    # création d'une grille de points de test
    r1 = np.arange(-5, 5, 0.2)
    Xtest = np.zeros((len(r1) * len(r1), 2))
    i = 0
    for x1 in r1:
        for x2 in r1:
            Xtest[i, :] = [x1, x2]
            i+=1

    # Prédire la catégorie pour tous les points de test...

    Ypred = modelsvm.predict(Xtest)

    # ... et tracer le résultat avec par exemple

    plt.scatter(Xtest[:,0],Xtest[:,1],c=Ypred)

    # tracer la droite séparation et les marges...

    w = modelsvm.coef_[0]
    b = modelsvm.intercept_
    m = 1 / np.sqrt(np.sum(w ** 2))
    xp = np.linspace(-5, 5)

    yp = -w[0] / w[1] * xp - b / w[1]
    plt.plot(xp,yp)
    yp_b =yp-np.sqrt(1-w[0]/w[1]**2)*m
    plt.plot(xp,yp_b)
    yp_h =yp+np.sqrt(1-w[0]/w[1]**2)*m
    plt.plot(xp, yp_h)

    # calculer et afficher la marge Delta...

    plt.fill_between(xp,yp_b,yp_h,alpha=0.4)

    # pour réellement mettre à jour le graphique: (à garder en fin de fonction)
    fig.canvas.draw()


def monprogrammeNL(Xapp, Yapp, C, sigma):
    """
		Programme pour les SVM non linéaires (lancé avec N)

		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
		sigma : paramètre réglé par CTRL +/-
	"""
    print("Apprentissage lancé avec " + str(len(Xapp)) + " points, C = ", C, " et sigma = ", sigma)

    # à compléter pour apprendre le modèle SVM non linéaire...

    model = svm.SVC(C=C, kernel="rbf", gamma=1/(2*sigma**2))
    model.fit(Xapp, Yapp)

    # création d'une grille de points de test
    r1 = np.arange(-5, 5, 0.2)
    Xtest = np.zeros((len(r1) * len(r1), 2))
    i = 0
    for x1 in r1:
        for x2 in r1:
            Xtest[i, :] = [x1, x2]
            i += 1

    # Prédire la catégorie pour tous les points de test...

    Ypred = model.predict(Xtest)

    # ... et tracer le résultat avec par exemple


    plt.scatter(Xtest[:,0],Xtest[:,1],c=Ypred)


    # afficher le nombre de vecteurs support...

    print("le nombre de vecteurs support",model.n_support_)

    # pour réellement mettre à jour le graphique:
    fig.canvas.draw()
    return model.n_support_

##### Gestion de l'interface graphique ########


Xplot = np.zeros((0, 2))
Yplot = np.zeros(0)
plotvariance = 0

C =math.inf #1
sigma = 1


def onclick(event):
    global Xplot
    global Yplot

    if plotvariance == 0:
        newX = np.array([[event.xdata, event.ydata]])
    else:
        newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10, 1)).dot(
            np.array([[event.xdata, event.ydata]]))

    print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

    Xplot = np.concatenate((Xplot, newX))
    if event.button == 1 and event.key == None:
        plt.plot(newX[:, 0], newX[:, 1], '.b')
        newY = np.ones(len(newX)) * 1
    elif event.button == 3 and event.key == None:
        plt.plot(newX[:, 0], newX[:, 1], '.r')
        newY = np.ones(len(newX)) * 2
    Yplot = np.concatenate((Yplot, newY))

    fig.canvas.draw()


def onscroll(event):
    global plotvariance
    if event.button == "up":
        plotvariance = round(plotvariance + 0.2, 1)
    elif event.button == "down" and plotvariance > 0.1:
        plotvariance = round(plotvariance - 0.2, 1)
    print("Variance = ", plotvariance)


def onkeypress(event):
    global C
    global sigma
    if event.key == " ":
        monprogramme(Xplot, Yplot, C)
    elif event.key == "n":
        monprogrammeNL(Xplot, Yplot, C, sigma)
    elif event.key == "$":#"+":
        C *= 2
        print("C = ", C)
    elif event.key == "*":#"-":
        C /= 2
        print("C = ", C)
    elif event.key == 'p':#"ctrl++":
        sigma *= 2
        print("sigma = ", sigma)
    elif event.key == 'o':#"ctrl+-":
        sigma /= 2
        print("sigma = ", sigma)


fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points bleus")
print(" clic droit : points rouges")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(Xapp,Yapp,C)")
print("    avec la valeur de C modifiée par +/-")
print("\n N pour lancer la fonction monprogrammeNL(Xapp,Yapp,C,sigma)")
print("    avec la valeur de C modifiée par +/-")
print("    et celle de sigma modifiée par CTRL +/-\n\n")

plt.show()

dict_col={1:'b',2:'r'}

if not os.path.exists("C_test"):
    os.makedirs("C_test")

c_values=[0.01,0.1,1,10,100,1000]

for c in c_values:
    C=c
    fig = plt.figure()
    plt.axis([-5, 5, -5, 5])
    col=[dict_col[i] for i in Yplot]
    plt.scatter(Xplot[:,0],Xplot[:,1],c=col)
    monprogramme(Xplot, Yplot, C)
    plt.savefig("C_test/C"+str(c)+".png")
    #plt.show(i)

if not os.path.exists("NL_C_test"):
    os.makedirs("NL_C_test")

num_vector=[]
for C in c_values:
    fig = plt.figure()
    plt.axis([-5, 5, -5, 5])
    col=[dict_col[i] for i in Yplot]
    plt.scatter(Xplot[:,0],Xplot[:,1],c=col)
    num_vector.append(monprogrammeNL(Xplot, Yplot, C, sigma))
    plt.savefig("NL_C_test/C_"+str(C)+".png")
    #plt.show(i)

fig = plt.figure("C_nbr_vector")
x = np.arange(len(c_values))
w=0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x- w/2,np.array(num_vector)[:,0], w, label='Class 1')
rects2 = ax.bar(x+ w/2,np.array(num_vector)[:,1], w, label='Class 2')
ax.set_ylabel('Nombre de vecteur support')
ax.set_xlabel('C')
ax.set_xticks(x)
ax.set_xticklabels(c_values)
ax.legend()
plt.savefig("C_nbr_vector.png")

C=1
sigma_values=[0.1,0.2,0.4,0.8,0.9,1]

if not os.path.exists("NL_sigma_test"):
    os.makedirs("NL_sigma_test")

num_vector=[]
for sigma in sigma_values:
    fig = plt.figure()
    plt.axis([-5, 5, -5, 5])
    col=[dict_col[i] for i in Yplot]
    plt.scatter(Xplot[:,0],Xplot[:,1],c=col)
    num_vector.append(monprogrammeNL(Xplot, Yplot, C, sigma))
    plt.savefig("NL_sigma_test/sigma_"+str(sigma)+".png")
    #plt.show(i)

fig = plt.figure("sigma_nbr_vector")
x = np.arange(len(sigma_values))
w=0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x- w/2,np.array(num_vector)[:,0], w, label='Class 1')
rects2 = ax.bar(x+ w/2,np.array(num_vector)[:,1], w, label='Class 2')
ax.set_ylabel('Nombre de vecteur support')
ax.set_xlabel('sigma')
ax.set_xticks(x)
ax.set_xticklabels(sigma_values)
ax.legend()
plt.savefig("sigma_nbr_vector.png")