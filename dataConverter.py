# This script converts the WKT format into following format used for GCMF code
# <polygon ID> <MBR> <vertices including closing vertex>
import pandas as pd
import re

def getMin(values, vNum, valX='x'):
    if valX=='X' or valX=='x':
        valX=0
    else:
        valX=1
    min=vertices[0].split(' ')[valX]    
    for i in range(1, vNum):
        if min>vertices[i].split(' ')[valX]:
            min=vertices[i].split(' ')[valX]
    return min

def getMax(vertices, vNum, valX='x'):
    if valX=='X' or valX=='x':
        valX=0
    else:
        valX=1
    max=vertices[0].split(' ')[valX]    
    for i in range(1, vNum):
        if max<vertices[i].split(' ')[valX]:
            max=vertices[i].split(' ')[valX]
    return max



# Use for ne_10m_ocean.csv
# file1="../datasets/ne_10m_ocean.csv"
# file2="../datasets/datasets/ne_10m_ocean.txt"
# df_file1=pd.read_csv (file1, header=None)
# f = open(file2, "a")
# pID=0
# df_results=[]
# polygons=pd.DataFrame()
# polygons=re.split('\(\(\(|\(', df_file1.loc[1, 0])
# pNum=len(polygons)
# print("Polygon count "+str(pNum))

# # testbed
# # tID=6823
# # print(polygons[tID])
# # vertices=re.split(",|\)", polygons[tID])
# # print("ii ")
# # print(vertices[0])

# for i in range(pNum):
#     # garbage value skip list
#     if i==0 or i==6806:
#         continue
#     print("Polygon "+str(pID)+" printing of "+str(i))
#     vertices=re.split(",|\)", polygons[i])
#     vNum=len(vertices)-3
#     # skip last polygon's split caused by ')))'
#     if i==len(polygons)-1:
#         vNum=len(vertices)-4
#     f.write("b "+str(pID)+" ")
#     f.write(str(getMin(vertices, vNum, 'x'))+" ")
#     f.write(str(getMin(vertices, vNum, 'y'))+" ")
#     f.write(str(getMax(vertices, vNum, 'x'))+" ")
#     f.write(str(getMax(vertices, vNum, 'y'))+" ")
#     for j in range(vNum):
#         f.write(str(vertices[j])+" ")
#     f.write(str(vertices[0]))
#     f.write('\n')
#     pID+=1

# f.close()

# Use for ne_10m_land.csv
file1="../datasets/ne_10m_land.csv"
file2="../datasets/datasets/ne_10m_land.txt"
df_file1=pd.read_csv (file1, header=None)
f = open(file2, "a")
pID=0
df_results=[]
polygons=pd.DataFrame()
# print(df_file1.loc[11, 0])
for k in range(1, 12):
    polygons=re.split('\(\(\(|\(\(|\(', df_file1.loc[k, 0])
    pNum=len(polygons)
    print("Polygon count "+str(pNum))

    # tID=2
    # # print(polygons[tID])
    # vertices=re.split(",|\)", polygons[tID])
    # print("ii ")
    # print(vertices[0])

    for i in range(pNum):
        # garbage value skip list
        if i==0 or i==6806:
            continue
        print("Polygon "+str(pID)+" printing of "+str(i))
        vertices=re.split(",|\)", polygons[i])
        vNum=len(vertices)-3
        # skip last polygon's split caused by ')))'
        if i==len(polygons)-1:
            vNum=len(vertices)-4
        f.write(str(pID)+" ")
        f.write(str(getMin(vertices, vNum, 'x'))+" ")
        f.write(str(getMin(vertices, vNum, 'y'))+" ")
        f.write(str(getMax(vertices, vNum, 'x'))+" ")
        f.write(str(getMax(vertices, vNum, 'y'))+" ")
        for j in range(vNum):
            f.write(str(vertices[j])+" ")
        if(vertices[0]!=vertices[vNum-1]):
            f.write(str(vertices[0])+" ")
        f.write('\n')
        pID+=1

f.close()