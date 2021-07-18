import time

if __name__ == '__main__':
    from collections import deque

    import global_var

    imgUrlQueue = deque()

    while True:
        imgUrl = "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg"
        imgUrlQueue.append(imgUrl)
        imgUrl = "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg"
        imgUrlQueue.append(imgUrl)
        imgUrlQueue.append(imgUrl)
        print(len(imgUrlQueue))
        global_var._setValue("img", imgUrlQueue)
        imgUrlQueue = global_var._getValue("img")
        time.sleep(5)
