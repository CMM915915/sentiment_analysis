#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:mengmeng
# 日期:2020/3/27
from multiprocessing import Process
from time import sleep

a = 1

def worker(sec,msg):
    #当worker作为子进程运行时,对全局量a 的修改只会
    #影响在子进程中a的值 ,对父进程没有影响
    global a
    a = 1000
    for i in range(3):
        sleep(sec)
        print("the worker msg",msg)
    print(a)

p = Process(name = 'worker',target = worker,args = (2,),kwargs = {'msg':'You are a big man'})

p.start()

#进程名称
print("进程名称:",p.name)
print("进程PID:",p.pid)
print('进程状态:',p.is_alive())

p.join()
print('parent:',a)