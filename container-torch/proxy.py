import os
import time
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from multiprocessing import Process

exec_path = './proxy/exec/'

proxy = Flask(__name__)
proxy.status = 'new'
proxy.debug = True


@proxy.route('/status', methods=['GET'])
def status():
    res = {}
    res['status'] = proxy.status
    res['workdir'] = os.getcwd()
    return res


@proxy.route('/url/<path:imgurl>', methods=['GET', 'POST'])
def imgUrl_predict(imgurl):
    from exec.network import getResult
    result = getResult(imgurl)
    return result


@proxy.route('/batch-url/<path:imgurl>', methods=['GET', 'POST'])
def batch_imgUrl_predict(imgurl):
    from exec.network import getResultList
    img = "https://dd-static.jd.com/ddimg/jfs/t1/160106/5/18226/60710/6075b218E6137405a/a4051ab3f28f536a.jpg"
    result = getResultList([imgurl, img] * 6)
    return result


@proxy.route('/light-model/<model>/url/<path:imgurl>', methods=['GET', 'POST'])
def light_model_imgUrl_predict(model, imgurl):
    from exec.network import getResultList_lightmodel
    result = getResultList_lightmodel(model, imgurl)
    return result


if __name__ == '__main__':
    server = WSGIServer(('0.0.0.0', 5000), proxy)
    server.serve_forever()
    # proxy.run()
