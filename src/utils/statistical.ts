import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';


export const randomNormal = (mean: number, std: number) => tf.tidy(() => tf.randomNormal([1], mean, std).get(0));
export const randomUniform = (min: number, max: number) => tf.tidy(() => tf.randomUniform([1], min, max).get(0));

export function softmax(arr: number[], a: number) {
    const sum = _.sum(arr.map(v => Math.exp(v)));
    return Math.exp(arr[a]) / sum;
}
