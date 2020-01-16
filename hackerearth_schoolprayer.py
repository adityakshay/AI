'''
# Sample code to perform I/O:

name = input()                  # Reading input from STDIN
print('Hi, %s.' % name)         # Writing output to STDOUT

# Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
'''
import numpy as np

def solve (students):
    # // for C users, use N = sizeof(students)/sizeof(students[0]) or
    # similar in other languages.
    #que = np.array()
    students = np.array(list(students))
    que = np.array([])
    allpos = np.array([])
   
    for i in range(len(students)):
        
        
       
        if i==0 or students[i] < min(que):
            pos = -1
            allpos=np.append(allpos,pos)
        else:
            if students[i] > min(que) and students[i] < max(que):
                for j in range(len(que)):
                    if que[j]<students[i] and que[j+1]>students[i]:
                        print(students[i])
                        print(len(que))
                        pos=que[j]
                        allpos=np.append(allpos, pos)
            else:
                pos=max(que)
                allpos=np.append(allpos, pos)
        que=np.append(que, students[i])
        que = np.sort(que)
        
    print(allpos)
    return(sum(allpos))
                    

T = int(input())

for _ in range(T):
    N = int(input())
    
    students = map(int, input().split())
    out_ = solve(students)
    print (out_)

