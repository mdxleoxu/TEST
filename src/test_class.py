import multiprocessing as mp



class A:
    def __init__(self):
        return
    

    def foo(self,i):
        print(i)
        return

    def mul_foo(self,i):
        pool = mp.Pool(mp.cpu_count())
        ret = []
        ret = [pool.apply_async(self.foo,args=(j)).get()[0] for j in range(i)]
        pool.close()
        print(ret)