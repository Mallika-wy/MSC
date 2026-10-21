"""Shared SONIC CUDA inference; simulation workers retain independent states."""
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from threading import Thread
from types import SimpleNamespace
import traceback

import numpy as np
import yaml

from data_pipeline.storage.store import digest, fingerprint


def batch_model(source, output, batch_size):
    """Adapt the official fixed-batch export without changing learned weights."""
    import onnx
    from onnx import numpy_helper
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    key = fingerprint([digest(source), digest(__file__), batch_size])
    target = output / f'{key}.onnx'
    if target.exists():
        return target
    model = onnx.load(source)
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = batch_size
    del model.graph.value_info[:]
    reshapes = {node.input[1] for node in model.graph.node if node.op_type == 'Reshape'}
    encoder = any(node.op_type == 'ScatterND' for node in model.graph.node)
    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        for attribute in node.attribute:
            if attribute.name != 'value':
                continue
            value = numpy_helper.to_array(attribute.t).copy()
            if node.output[0] in reshapes and value.ndim == 1 and len(value) > 1 and value[0] == 1:
                value[0] = batch_size
            elif encoder and node.name == '/Constant_1':
                value = np.arange(batch_size, dtype=np.int64)
            elif encoder and node.name == '/Constant_3':
                value = np.zeros((batch_size, 3), dtype=np.float32)
            else:
                continue
            attribute.t.CopyFrom(numpy_helper.from_array(value))
    onnx.save(model, target)
    return target


class PolicyClient:
    def __init__(self, kind, slot, state):
        inputs, outputs, self.events, self.requests, _, self.failed = state
        self.kind, self.slot = kind, slot
        self.inputs = np.frombuffer(inputs, dtype=np.float32).reshape(-1, 1751)
        self.outputs = np.frombuffer(outputs, dtype=np.float32).reshape(-1, 64)

    def get_inputs(self):
        return [SimpleNamespace(name='obs_dict')]

    def run(self, names, feed):
        if self.failed.is_set():
            raise RuntimeError('Shared SONIC GPU inference failed')
        values = feed['obs_dict']
        self.inputs[self.slot, :values.shape[1]] = values[0]
        event = self.events[self.slot]
        event.clear()
        self.requests.put((self.kind, self.slot))
        if not event.wait(120) or self.failed.is_set():
            raise RuntimeError('Shared SONIC GPU inference failed or timed out')
        width = 64 if self.kind == 0 else 29
        return [self.outputs[self.slot:self.slot+1, :width].copy()]


def connect_worker(state, observation_config):
    with state[4].get_lock():
        slot = state[4].value
        state[4].value += 1
    config = yaml.safe_load(Path(observation_config).read_text())
    return PolicyClient(0, slot, state), PolicyClient(1, slot, state), config


class SharedPolicy:
    """One parent thread serves all simulation processes on one CUDA device."""
    def __init__(self, execution, workers, output):
        import torch  # Make the installed CUDA/cuDNN libraries available to ORT.
        import onnxruntime as ort
        ort.preload_dlls()
        self.batch_size = execution['inference_batch_size']
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        providers = [('CUDAExecutionProvider', {'use_tf32': False,
                      'device_id': execution.get('device_id', 0)})]
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('Shared SONIC inference requires onnxruntime-gpu')
        self.sessions = [ort.InferenceSession(str(batch_model(execution[key], output, self.batch_size)),
                         options, providers=providers) for key in ('encoder', 'decoder')]
        if any(session.get_providers()[0] != 'CUDAExecutionProvider' for session in self.sessions):
            raise RuntimeError('SONIC could not initialize CUDAExecutionProvider')
        ctx = get_context('spawn')
        self.requests = ctx.Queue()
        self.events = [ctx.Event() for _ in range(workers)]
        self.failed = ctx.Event()
        inputs, outputs = ctx.RawArray('f', workers*1751), ctx.RawArray('f', workers*64)
        self.worker_state = inputs, outputs, self.events, self.requests, ctx.Value('i', 0), self.failed
        self.inputs = np.frombuffer(inputs, dtype=np.float32).reshape(workers, 1751)
        self.outputs = np.frombuffer(outputs, dtype=np.float32).reshape(workers, 64)
        self.stopped = False
        self.thread = Thread(target=self._serve, name='sonic-gpu', daemon=True)

    def _serve(self):
        pending = [[], []]
        try:
            while not self.stopped:
                if not any(pending):
                    try:
                        kind, slot = self.requests.get(timeout=.1)
                        pending[kind].append(slot)
                    except Empty:
                        continue
                while sum(map(len, pending)) < 2*self.batch_size:
                    try:
                        kind, slot = self.requests.get_nowait()
                        pending[kind].append(slot)
                    except Empty:
                        break
                kind = 0 if len(pending[0]) >= len(pending[1]) else 1
                slots = pending[kind][:self.batch_size]
                del pending[kind][:self.batch_size]
                width = 1751 if kind == 0 else 994
                batch = np.zeros((self.batch_size, width), dtype=np.float32)
                batch[:len(slots)] = self.inputs[slots, :width]
                result = self.sessions[kind].run(None, {'obs_dict': batch})[0]
                self.outputs[slots, :result.shape[1]] = result[:len(slots)]
                for slot in slots:
                    self.events[slot].set()
        except BaseException:
            traceback.print_exc()
            self.failed.set()
            for event in self.events:
                event.set()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped = True
        self.thread.join()
        self.requests.close()
        self.requests.join_thread()
