import time
from time import sleep

def smart_cache(lru_cache_size,lifetime):
    results=dict()
    counter = 1
    popa = 1
    def wrapper(func):
        def inner(*args,**kwargs):
            nonlocal counter,results,popa
            a = func(*args,**kwargs)
            if len(results)<lru_cache_size:
                if len(results)>0:
                    if a!=results[counter-1][0]:
                        pop_keys1 = []
                        for key in results:
                                pop_keys1.append(key)
                        for i in pop_keys1:
                            results.pop(i)
                results[counter] = [a, time.time()]
                counter+=1
            else:
                results.pop(popa)
                popa+=1
            pop_keys=[]
            for key in results:
                if time.time() - results[key][1]>= lifetime:
                    pop_keys.append(key)
            for i in pop_keys:
                results.pop(i)
            for key in results:
                print('Test number: '+ str(key)+'  result is: '+str(results[key][0]))
        return inner
    return wrapper

@smart_cache(2, 3)
def my_func1(a,b):
    print('func1 is working')
    return a+b


@smart_cache(10, 3)
def my_func2(a,b):
    print('func2 is working')
    return a


my_func1(1,1)
time.sleep(1)
my_func1(1,1)
time.sleep(1)
my_func1(1,1)
time.sleep(1)
my_func1(2,1)