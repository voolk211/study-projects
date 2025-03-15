import sys

name = 'tata2025.tsv' #sys.argv[1]
player=1
try:
    player=sys.argv[2]
except:
    pass
file=open(name,'r',encoding='utf8')
s = []
while True:
    try:
        a = file.readline()
        if(a==''):
            break
        s.append(a.split('\t'))
    except:
        break

dict={}

for i in range(0,len(s),1):
    dict[s[i][0]] = s[i][2]

print(dict)