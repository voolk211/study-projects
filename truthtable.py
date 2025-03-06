
n=50
s=[i for i in range(2,n+1)]
answ=[]
for i in range(0,len(s)):
    if (s[i]!=0):
        answ.append(s[i])
    tmp=s[i]
    j=i
    if(tmp!=0):
        while(j<len(s)):
            s[j]=0;
            j+=tmp
print(answ)