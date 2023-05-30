f = open("planning_KB.txt",encoding='utf-8')
count = 0
while True:
    line = f.readline()
    if line:
        count += 1
    else:
        break
print(count)
f.close()