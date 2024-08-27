# -*- coding: utf-8 -*-

      #% matplotlib inline
import numpy as np
import matplotlib . pyplot as plt
from scipy . integrate import solve_ivp
import os , json
import myeoslibs
from sympy import *

# Define right - hand system of equations
def tovrhs (r,z):
    M = z [0] # mass coordinate as part of array z
    P = z [1] # pressure coordinate as part of array z
    y = z [2] #yr parameter
    if P0 >= 0.184:
        e = myeoslibs.eoslib[ eostype ](P) #Equation of state
        dd= deos(P)
    elif P0 >= 9.34375e-5:
        e= myeoslibs.eoslib['eoscore'](P) #1st crust equation
        dd= deoscore(P)
    elif P0 >= 4.1725e-8:
        e= myeoslibs.eoslib['eoscore2'](P) #2nd crust equation
        dd= deoscore2(P)
    elif P0 >= 1.44875e-11:
        e= myeoslibs.eoslib['eoscore3'](P) #3rd crust equation
        dd= deoscore3(P)
    else :
        e= myeoslibs.eoslib['eoscore4'](P) #4th crust equation
        dd= deoscore4(P)
    dMdt = 11.2 * (10 ** (-6) )* (r** 2)* e #1st order derivative of Mass
    dPdt = - 1.474 * (e* M/( r** 2) )* (1+P/e )* (1+ 11.2 * (10 ** (-6) )* (r** 3)* P/M )* ((1 -2.948 * M/r )** (-1) ) #1st order derivative of Pressure
    F = (1- 1.474 * 11.2 * (10 ** (-6) )* (r** 2)* (e-P))* ((1 - 2.948 * M/r )** (-1) ) #F part
    J = 1.474 * 11.2 * (10 ** (-6) )* (r** 2)* (5* e+9* P+(( e+P )/(1/dd )))* ((1 - 2.948 * M/r )** (-1) )-6* ((1 - 2.948 * M/r )** (-1) )-4* ((1.474 ** 2)* (M** 2)/(r** 2) )* ((1 + 11.2 *(10 ** (-6) )* (r** 3)* (P/M))** 2)* ((1 - 2.948 * M/r )** (-2) )
    dyrdt = (-y* y-y* F-J )/r
    dzdt = [dMdt ,dPdt , dyrdt ] # array of derivatives
    return dzdt # return the array of derivatives

# initial conditions
Step =0.0001;
G = 6.674*10**(- 33) ;
n =100;
ic1 = np.arange(1.5 ,5 ,0.1)
ic2 = np.arange(5 ,1201 ,1)
ic=np.concatenate(( ic1 , ic2 ),axis = None )

# insert EOS and crust EOS
eostype = 'lattimer_soft'
pp = Symbol('pp')
eosfuncprime = myeoslibs.eossym[ eostype ]().diff(pp)
deos = lambdify (pp , eosfuncprime ,'numpy')

eosfuncprimecore = myeoslibs.eossym['eoscore']().diff( pp )
deoscore = lambdify( pp , eosfuncprimecore ,'numpy')
eosfuncprimecore2 = myeoslibs.eossym['eoscore2']().diff( pp )
deoscore2 = lambdify( pp , eosfuncprimecore2 ,'numpy')
eosfuncprimecore3 = myeoslibs.eossym['eoscore3']().diff( pp )
deoscore3 = lambdify( pp , eosfuncprimecore3 ,'numpy')
eosfuncprimecore4 = myeoslibs.eossym['eoscore4']().diff( pp )
deoscore4 = lambdify( pp , eosfuncprimecore4 ,'numpy')

# setting up working directory
directory = os.path.join( os.getcwd() , eostype )
if not os.path.exists( directory ):
    print(" path doesn 't exist.trying to make ")
    os.makedirs( directory )


 ################################# SOLVING #####################################

minmax = np.zeros (( len( ic ) , 6) ) ;
j = 0;

for i in ic :
    z0 =[0.000000000001 , i ,2.]
    P0 = 1.0;
    rmin =0.00000001;
    rmax =.01;
    Mf = np.array ([])
    Pf = np.array ([])
    R = np.array ([])
    yr = np.array ([])
    z0_old = np.array ([])
    print(i)
    while ( P0>1e-12) :
        res = solve_ivp ( tovrhs ,( rmin , rmax ) ,z0 , method ='LSODA', atol =10 ** -26 , rtol=10 ** -8)
       # print(res.y[0][-1])
        z0_old = z0 [1]
        z0 [1]= res.y [1][~np.isnan ( res.y [1]) ][-1]
        z0 [0]= res.y [0][~np.isnan ( res.y [0]) ][-1]
        z0 [2]= res.y [2][~np.isnan ( res.y [2]) ][-1]
        if z0 [0]<0:
            break
        if ( z0_old == z0 [1]) :
            break
        rmin = res.t[~np.isnan ( res.y [2]) ][-1]
        rmax = rmin + 0.001
        P0 = z0 [1]
        Mf = np.append (Mf , res.y [0][~np.isnan ( res.y [0]) ])
        Pf = np.append (Pf , res.y [1][~np.isnan ( res.y [1]) ])
        yr = np.append (yr , res.y [2][~np.isnan ( res.y [2]) ])
        R = np.append (R , res.t[~np.isnan ( res.y [2]) ])
    if Pf [-1]<0:
        idx = np.argwhere ( Pf<0) [0 ,0]
        Pf = np.delete (Pf , np.s_ [ idx ::] , 0)
        Mf = np.delete (Mf , np.s_ [ idx ::] , 0)
        yr = np.delete (yr , np.s_ [ idx ::] , 0)
        R = np.delete (R , np.s_ [ idx ::] , 0)
    minmax [ j] = R[-1] , max ( Mf ) ,min ( Pf ) , yr [-1] ,0 ,0
    j = j + 1;
    beta = 1.474 * minmax [ j- 1][1] / minmax [j- 1][0]
    k2 = (8 * ( beta ** 5)/5) * ((1 -2* beta )** 2)* (2-yr [-1]+2* beta * ( yr [-1]-1) )* (2*beta * (6-3* yr [-1]+3* beta * (5* yr [-1]-8) )+4* ( beta ** 3)* (13 -11 * yr [-1]+ beta * (3*
    yr [-1]-2)+2* ( beta ** 2)* (1+yr [-1]) )+3* ((1 -2* beta )** 2)* (2-yr [-1]+2* beta * ( yr[-1]-1) )* np . log (1-2* beta ))** (-1)
    l = 2/3* (( R[-1] ** 5)/G)* k2* 10** (- 36)
    minmax [ j- 1][4] = k2
    minmax [ j- 1][5] = l

with open ( os . path . join ( directory ,'data_'+ eostype +'.txt') , mode ='w') as fl :
    namefile ='---- data for EoS:'+ eostype +'\r\n'
    fl . write ( namefile )
with open( os.path.join( directory ,'data_'+ eostype +'.txt') , mode ='ab') as fl :
    np . savetxt ( fl , minmax , delimiter =",", header ='radius R (km)\t\t mass M (Msun )\t\t pressure P \t yr \t\t k2 \t\t lambda ', fmt ="%.16f", comments ='', newline ='\r\n')
with open ( os . path . join ( directory ,'data_'+ eostype +'.json') , 'w') as outfile :
    json . dump ( minmax . tolist () , outfile )

