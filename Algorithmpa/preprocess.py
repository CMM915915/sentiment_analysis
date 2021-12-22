#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:mengmeng
# 日期:2020/3/27

import random
import time
import multiprocessing


def worker(name, q):
    t = 0
    for i in range(10):
        print(name + " " + str(i))
        x = random.randint(1, 3)
        t += x
        time.sleep(x * 0.1)
    q.put(t)

if __name__=='__main__':
    q = multiprocessing.Queue()
    jobs = []
    for i in range(10):
        p = multiprocessing.Process(target=worker, args=(str(i), q))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = [q.get() for j in jobs]
    print(results)

