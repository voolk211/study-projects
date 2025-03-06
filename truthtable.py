
n=5
nn=n-1
values=[1]*n
values_=[1]*n
for i in range(1,n+1):
    values[nn]=2**i//2
    values_[nn]=-values[nn]
    nn-=1
znachenia=values.copy()

length=2**n
for i in range(0,length):
    s=[0]*n
    for j in range(0,n):
        if(znachenia[j]>0):
            s[j]=1
            znachenia[j]-=1
        else:
            s[j] = 0
            znachenia[j] -= 1
            if (znachenia[j]==values_[j]):
                znachenia[j]=values[j]
    print(s)

