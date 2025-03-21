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
        s.append(a.strip('\n').split('\t'))
    except:
        break

dictn = {}
for i in range(0,len(s),1):
        if s[i][0] not in dictn:
            dictn[s[i][0]] = [[0,0,0,0,0.,0.],set(),set()]



        dictn[s[i][0]][0][0] = dictn[s[i][0]][0][0]+1 #GAMES

        if float(s[i][2]) == 1.:
            dictn[s[i][0]][0][1] = dictn[s[i][0]][0][1] + 1 #WON
            dictn[s[i][0]][1].add(s[i][1])
            #print(dictn[s[i][0]][1])

        elif float(s[i][2]) == 0.:
            dictn[s[i][0]][0][3] = dictn[s[i][0]][0][3] + 1 #LOSE
        else: #float(s[i][2]) == 0.5:
            dictn[s[i][0]][0][2] = dictn[s[i][0]][0][2] + 1 #DRAWN
            dictn[s[i][0]][2].add(s[i][1])

        dictn[s[i][0]][0][4] =  dictn[s[i][0]][0][4] + float(s[i][2]) #SCORE


        if s[i][1] not in dictn:
            dictn[s[i][1]] = [[0,0,0,0,0.,0.],set(),set()]

        dictn[s[i][1]][0][0] = dictn[s[i][1]][0][0] + 1 #GAMES

        if float(s[i][3]) == 1.:
            dictn[s[i][1]][0][1] = dictn[s[i][1]][0][1] + 1 #WON
            dictn[s[i][1]][1].add(s[i][0])

        elif float(s[i][3]) == 0.:
            dictn[s[i][1]][0][3] = dictn[s[i][1]][0][3] + 1 #LOSE
        else:  # float(s[i][3]) == 0.5:
            dictn[s[i][1]][0][2] = dictn[s[i][1]][0][2] + 1 #DRAWN
            dictn[s[i][1]][2].add(s[i][0])

        dictn[s[i][1]][0][4] = dictn[s[i][1]][0][4] + float(s[i][3]) #SCORE


        #if (i!=len(s)-1):
        #    dictn[s[i][1]] = float(s[i][3][:-1])
        #else:dictn[s[i][1]] = float(s[i][3])
for key in dictn:
    #print(dictn[key])
    sb1=0
    for i in dictn[key][1]:
        #print(i)
        #print(dictn[i][0][4])
        sb1=sb1+dictn[i][0][4]

    for i in dictn[key][2]:
        sb1 = sb1 + (dictn[i][0][4])/2
    dictn[key][0][5] = sb1

tournament_rating=[]
k=1
for key in dictn:
    tournament_rating.append([k,key,dictn[key][0]])
    k+=1

for i in range(len(tournament_rating)-1):


print(tournament_rating)

