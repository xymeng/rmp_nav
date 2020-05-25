import zmq
import sys
import gflags
import time
import multiprocessing
import threading
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
from .inference import find_controller


gflags.DEFINE_string('addr', '', '')
gflags.DEFINE_string('weights_file', '', '')
gflags.DEFINE_string('class_path', '', 'If specified will use this class to load weights file.')
gflags.DEFINE_integer('n_worker', 1, '')


FLAGS = gflags.FLAGS
FLAGS(sys.argv)


def backend_process(listen_addr):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(listen_addr)

    if FLAGS.class_path != '':
        print('inference server class path:', FLAGS.class_path)
        # class_path is of the form: package1.[package2...].class_name
        toks = FLAGS.class_path.split('.')
        module_path = '.'.join(toks[:-1])
        class_name = toks[-1]
        mod = __import__(module_path, fromlist=[class_name])
        mp = getattr(mod, class_name)(FLAGS.weights_file)
    else:
        mp = find_controller(FLAGS.weights_file)(FLAGS.weights_file)

    def exec_cmd(msg):
        func_name = msg[0]
        args = msg[1:]
        try:
            ret = getattr(mp, func_name)(*args)
            return ret
        except:
            print('Failed to execute message: %r' % msg)
            raise

    while True:
        msg = msgpack.unpackb(socket.recv(), raw=False)
        ret = None

        if msg[0] == 'exit':
            socket.send_pyobj('ok')
            break

        elif msg[0] == 'cmd_list':
            for m in msg[1:]:
                ret = exec_cmd(m)
        else:
            ret = exec_cmd(msg)

        socket.send(msgpack.packb(ret, use_bin_type=True))

    socket.disconnect(listen_addr)
    socket.close()
    context.term()
    print('motion policy inference server backend terminated')


if __name__ == '__main__':
    context = zmq.Context()

    frontend = context.socket(zmq.ROUTER)
    frontend.setsockopt(zmq.RCVHWM, 0)
    frontend.setsockopt(zmq.SNDHWM, 0)
    frontend.setsockopt(zmq.LINGER, 0)
    frontend.bind(FLAGS.addr)

    backend = context.socket(zmq.DEALER)
    backend.setsockopt(zmq.RCVHWM, 0)
    backend.setsockopt(zmq.SNDHWM, 0)
    backend.setsockopt(zmq.LINGER, 0)

    backend_addr = 'ipc:///tmp/motion_policy_inference-backend-%s' % str(time.time())
    backend.bind(backend_addr)

    backend_procs = []

    for i in range(FLAGS.n_worker):
        proc = multiprocessing.Process(target=backend_process, args=(backend_addr,))
        proc.start()
        backend_procs.append(proc)

    def monitor_backend():
        print('monitor thread started')
        for proc in backend_procs:
            proc.join()
        print('all backend procs terminated')
        context.term()

    monitor_thread = threading.Thread(target=monitor_backend)
    monitor_thread.start()

    try:
        zmq.device(zmq.QUEUE, frontend, backend)
    except zmq.error.ContextTerminated:
        frontend.close()
        backend.close()
        print('motion policy inference server frontend terminated')
