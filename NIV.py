import numpy as np
import scipy.stats as stats

# Observations from 6 leveling runs 1,2,3,4,5,6
OBS1 = -4.65022 #A1-B1 
OBS2 = -4.49775 # A1- S2 
OBS3 = -3.81794 #A1-S3 
OBS4 = 0.1673  #B1-S2
OBS5 = 0.83042 #S3-B1 =  
OBS6 = 0.66935  #S2-S3 

#Matrise og diag for vekting 
OBS = np.matrix([OBS1,OBS2,OBS3,OBS4,OBS5,OBS6])
OBS_D = np.matrix(np.diag(OBS.A1)) 

# Known point
known_point = 187.739
#Ass.point
B1C = known_point + OBS1 #A1-B1
S2C = known_point + OBS2 #A1-S2
S3C = known_point + OBS3 #A1-S3

#L Matrix
V1 = B1C - known_point - OBS1
V2 = S2C - known_point - OBS2
V3 = S3C - known_point - OBS3
V4 = S2C - B1C - OBS4
V5 = -B1C + S3C - OBS5
V6 = S3C - S2C - OBS6 


err = 0.002  #forventet feil 
lengde = np.matrix([0.532344,0.7931095,0.5626585,0.372011,0.1238,0.23299])#OBS lengde 1,2,3,4,5,6 i KM 
lengde_m = np.dot(lengde,err)# Lengde*err 
lengde_m2 = np.power(lengde_m.A1,2)
P = np.matrix(np.diag(np.divide(1,lengde_m2)))

#A Matrise (dh_x)
A = np.matrix([[1,0,0],[0,1,0],[0,0,1],[-1,1,0],[1,0,-1],[0,-1,1]])
AT = A.T
#L Matrise 
L = np.matrix([V1,V2,V3,V4,V5,V6]).T

# Solving the normal equation… 
x_1 = AT * P * A
x_2 = AT * P * L
x = x_1.I * x_2

#Applying the x solutions to the observation equations…
V = A * x -L
R = np.dot(np.dot(P, V), V.T)


#Correcting observations 
OBS_KOR = np.array([OBS.A1 + V.A1])
OBS_KOR = OBS_KOR.flatten()

#Corrected Heights 
B1C_K = known_point + OBS_KOR[0] #A1-B1
S2C_K = known_point + OBS_KOR[1] #A1-S2
S3C_K = known_point + OBS_KOR[2] #A1-S3


#*Standard deviation of the unknown parameters*#
#Cofactor Matrix
Qxx = np.linalg.inv(A.T @ P @ A)
Quu=A*Qxx*A.T
SUMuu = R.item(0,0) * Quu
#Covariance matrix
SUMxx = R.item(0,0) * np.linalg.inv(A.T @ P @ A)
Vxx = np.diagonal(Qxx)
STDxx = np.sqrt(np.diagonal(np.linalg.inv(A.T @ P @ A)))
#STDxx = np.reshape(STDxx, (3,1))
##Standard deviation of the adjusted observations
Vuu = np.diagonal(Quu)
STDuu = np.sqrt(Vuu)

#Standard deviation of the residuals
Qvv = OBS_D.T*Quu*OBS_D 
SUMvv = R.item(0,0) * Qvv
Vvv = np.diagonal(Qvv)
STDvv = np.sqrt(Vvv)


#Chi squared test 
n = 6 # number of observations
m = 3 # number of unknown parameters
chi_squared = R.item(0, 0) / (n - m)
p_value = 1 - stats.chi2.cdf(chi_squared, n - m)
if p_value < 0.05:
    print("Chi Test: Reject the null hypothesis: The model does not fit the data well")
else:
    print("Chi Test: Fail to reject the null hypothesis: The model fits the data well")
    
    t, p = stats.ttest_ind(OBS.A1, OBS_KOR, equal_var=False)
print("t-statistic:", t)


#Remove hashtag to print 

#print("p-value:", p)
#print (A) # A Matrise 
#print(np.resize(np.matrix(np.divide(1,lengde_m2)), [6, 1])) ## P Matrise 
#print (L) #L Konstant matrise 
#print (x) #𝒙 vector (vector of the unknown parameters; corrections for the preliminary elevations) 
#print(V) #𝒗 vector (residuals) 
#print (np.resize(OBS_KOR,[6,1])) #Corrected Observations 
#print (" B1",B1C_K,'\n','S2',S2C_K,'\n',"S3",S2C_K) #Corrected Heights
#print('𝒗𝑻∙𝑷∙𝒗', R.item(0, 0)) #(weighted sum of the squares of the residuals) 
#print("𝒔^𝟐_𝟎", R.item(0, 0)/2) #DENNE ER VIKTIG 
#print(np.reshape(STDxx, [3,1])) # extracting the variances from the diagonal and processing their square roots:
#print (np.reshape(STDxx, (3,1))) #DENNE ER VIKTIG  UKNOWN PARAMETERS
#print(np.reshape(STDuu,(6,1))) #Standard deviation of the adjusted observations
#print(np.reshape(STDvv,(6,1))) # Standard deviation of the residuals 



