import multiprocessing as mp
 
def job(a,d):
    filter = [0,1,2,3,4,5,6,7,8,9]
    fitness = a*d
    
    print(fitness)
    
    print(filter[a])
    print('aaaaa')
 
if __name__=='__main__':
    p1 = mp.Process(target=job,args=(1,2))
    p1.start()
    p1.join()