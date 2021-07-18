# 使用 os 召唤 docker
# 创建和删除容器
import os
import time


def create_new_container(id):
    print("创建 {} 号容器".format(id))
    container_name = "hanwen_torch_{}".format(id)
    host_port_for_container = "500{}".format(id)
    host_path_for_container = "/Users/starky/PycharmProjects/docker-torch/storage/hanwen_torch_{}".format(id)
    # image_name = "hanwen_torch_base_run_inside_import"
    image_name = "hanwen_latest_cnn_2"

    # 复制文件
    # src = "/Users/starky/PycharmProjects/docker-torch/copy-code/image-classification"
    # src = "/Users/starky/PycharmProjects/docker-torch/copy-code/image-classification_new" # 正常运行跑这个就行
    # src = "/Users/starky/PycharmProjects/docker-torch/copy-code/image-classification_new_cache"
    src = "/Users/starky/PycharmProjects/docker-torch/copy-code/image-classification_process"  # 这个用于测试先下载的方式
    dst = host_path_for_container
    copy_command = "cp -r {} {}".format(src, dst)
    print(copy_command)

    time_before_copy = time.time()
    result = os.popen(copy_command)
    print(result.readlines())
    print("Copy uses {} s".format(time.time() - time_before_copy))

    print()

    # 创建新容器
    docker_command = "docker run -d --name {} -p {}:5000 -v {}:/proxy/exec {}".format(container_name,
                                                                                      host_port_for_container,
                                                                                      host_path_for_container,
                                                                                      image_name)
    print(docker_command)
    time_before_create_container = time.time()
    result = os.popen(docker_command)
    print(result.readlines())
    print("Create container uses {} s".format(time.time() - time_before_create_container))


def stop_and_remove_container(id):
    print("删除 {} 号容器".format(id))
    time_before_stop_and_remove_container = time.time()
    result = os.popen("docker stop hanwen_torch_{} && docker rm hanwen_torch_{}".format(id, id))
    print(result.readlines())

    host_port_for_container = "500{}".format(id)
    host_path_for_container = "/Users/starky/PycharmProjects/docker-torch/storage/hanwen_torch_{}".format(id)
    result = os.popen("rm -rf {}".format(host_path_for_container))
    print(result.readlines())
    print("Remove container uses {} s".format(time.time() - time_before_stop_and_remove_container))


if __name__ == '__main__':
    for i in range(1):
        stop_and_remove_container(i)
        # create_new_container(i)

    # create_new_container(4)
    # stop_and_remove_container(0)
    # stop_and_remove_container(4)

    # stop_and_remove_container(2)
    # stop_and_remove_container(3)
    # stop_and_remove_container(4)
