import random
import numpy as np

def sequencegenerator(length=16,sample='acgt'):
    s1 = ''.join((random.choice(sample) for x in range(length)))
    s2 = ''.join((random.choice(sample) for x in range(length)))
    return s1,s2



def max(a,b,c):
    if(a>b and a>c):
        return a
    elif(b>a and b>c):
        return b
    else:
        return c
    
    
def maxmatrix(c):
    d=0
    maxrow=maxcol=0
    for i in range(17):
        for j in range(17):
            if(c[i][j]>d):
                d=c[i][j]
                maxrow=i
                maxcol=j
    return d,maxrow,maxcol
                
def backtrack(n, c, maxrow, maxcol, a, b):
    
    if(n<0):
         return 
    else:
        print(c[maxrow][maxcol])
        if(a[maxrow-1]==b[maxcol-1]):
            return backtrack(n-1,c,maxrow-1,maxcol-1,a,b)
        
        else: 
            if(c[maxrow-1][maxcol-1]>c[maxrow-1][maxcol] and c[maxrow-1][maxcol-1]>c[maxrow][maxcol-1]):
                return backtrack(n-1,c,maxrow-1,maxcol-1,a,b)

            elif(c[maxrow-1][maxcol]>c[maxrow-1][maxcol-1] and c[maxrow-1][maxcol]>c[maxrow][maxcol-1]):
                return backtrack(n-1,c,maxrow-1,maxcol,a,b)
            
            else:
                return backtrack(n-1,c,maxrow,maxcol-1,a,b)
    


s1,s2=sequencegenerator()

s1=list(s1)
s2=list(s2)


matrix=np.zeros((17,17))

        


for i in range(16):
    for j in range(16):
        if(s1[i]==s2[j]):
            matrix[i+1][j+1]=matrix[i][j]+5
        else:
            m=max(matrix[i][j],matrix[i+1][j],matrix[i][j+1])
            matrix[i+1][j+1]=m-4
            

for i in range(17):
    for j in range(17):
        print(matrix[i][j])
    print()
    

maxval,maxrow,maxcol=maxmatrix(matrix)
backtrack(16,matrix,maxrow,maxcol,s1,s2)
    


