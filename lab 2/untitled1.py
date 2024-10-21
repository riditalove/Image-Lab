import random
import math

p = 252829
q = 442327

n = p*q
phi = (p-1)*(q-1)
lamda = phi/math.gcd(p-1,q-1)
if(lamda == int(lamda)):
    print("lamba is int")
g = random.randint(1,n)
lamda = int(lamda)
#while(1):
    #x = pow(g,lamda,n**2)
    #lx = (x-1)/n
    #if(lx == int(lx) and math.gcd(int(lx),n)==1):
        #break
   # else:
# g = random.randint(1, n)

glamda = pow(g,lamda,n**2)
L = (glamda-1)//n
miu = pow(L,-1,n)
m=1807000
r=11
c=pow(g,m,n**2)*pow(r,n,n**2)%(n**2)
print("c = ",c)
print("g = ",g)
print("lamda = ",lamda)
dec=((pow(c,lamda,n**2)-1)//n)*miu%n
print("decrypted message = ",dec)
    