import * as tf from '@tensorflow/tfjs';

export function updateAtIndex(vec: tf.Tensor1D, index: number, value: number): tf.Tensor1D {
    const buff = vec.buffer();
    buff.set(value, index);
    return buff.toTensor();
}
