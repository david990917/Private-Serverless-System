# 对于cat/dog文件读取的方式测试
ff = open("url_idx.txt","w")
with open("/Users/starky/PycharmProjects/docker-torch/cat_1000.txt") as f:
    lines = f.readlines()
    for line in lines:
        ff.write(line)

with open("/Users/starky/PycharmProjects/docker-torch/dog_1000.txt") as f:
    lines = f.readlines()
    for line in lines:
        ff.write(line)

ff.close()
