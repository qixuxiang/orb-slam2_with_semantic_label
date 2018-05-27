# coding=utf-8
import os

input_ = open('1.txt', 'r')
output  = open('src.txt', 'w')

for line in input_:
    output.write('${DARKNET_PATH}/src/'+ line)
    
output.close()
input_.close()