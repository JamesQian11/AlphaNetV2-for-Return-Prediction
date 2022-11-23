import os

path = 'zz500'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)

print(fileList)
n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
    print(fileList[n].split('_'))
    # # 设置新文件名
    a = fileList[n].split('_')[0] + '_' + 'zz500' + '_' + fileList[n].split('_')[2] + '_' + fileList[n].split('_')[3]
    print(a)
    newname = path + os.sep + a
    #
    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    # print(oldname, '======>', newname)

    n += 1
