# 使用 docker 库来召唤 docker
import docker

client = docker.from_env()

images_list = client.images.list()  # [<Image: 'hanwen_torch_base:latest'>]

# container.run 不是很灵光
# container = client.containers.run("hanwen_torch_base","python3 /proxy/proxy.py",
#                                   name="hanwen_torch_3",
#                                   detach=True,
#                                   ports={'5000': 5003},
#                                   volumes={
#                                       "//Users/starky/PycharmProjects/docker-torch/storage/hanwen_torch_3":
#                                           {'bind': "/proxy",
#                                            'mode': "rw"},
#                                   })
# print(container.name)
# print(container.id)

containers_list = client.containers.list()
containers_name_list = [container.name for container in containers_list]
print(containers_list)
print(containers_name_list)

for container in containers_list:
    print(container.name)
    container.stop()