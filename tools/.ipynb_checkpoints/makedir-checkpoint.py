import os
def makedir(root):
    # print(root)
    filelist=os.listdir(root)
    # print(filelist)
    if len(filelist)==0:
        name='0'
        path = root + name
        os.mkdir(path)
        return path
    tonum=[]
    for it in filelist:
        tonum.append(int(it))
    tonum.sort()
    name=str(tonum[len(tonum) - 1] + 1)
    path=root+name
    os.mkdir(path)
    return path

if __name__=="__main__":
    root="./runs/"
    name=makedir(root)
    print(name)
