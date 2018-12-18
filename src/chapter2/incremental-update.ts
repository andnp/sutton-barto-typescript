import * as tf from '@tensorflow/tfjs';
import * as _ from 'lodash';
import { random } from 'mlts';
import { Matrix } from 'utilities-ts';
import { describeColumns, LineChart, Remote, createStandardPalette, combineTraces } from 'tsplot';
import '@tensorflow/tfjs-node';

class KArmedBandit {
    private q_star: tf.Tensor1D;
    constructor(private k: number) {
        this.q_star = tf.randomNormal([k], 0, 1);
    }

    takeAction(a: number): number {
        if (a < 0 || a >= this.k) throw new Error(`Action is outside of range. ${a}`);

        const value = this.q_star.get(a);
        const reward = tf.randomNormal([1], value, 1).asScalar();
        return reward.get();
    }
}

const argMax = (arr: number[]): number => {
    let m = Number.NEGATIVE_INFINITY;
    let locs = [] as number[];
    for (let i = 0; i < arr.length; ++i) {
        if (arr[i] === m) {
            locs.push(i);
        } else if (arr[i] > m) {
            locs = [i];
            m = arr[i];
        }
    }

    if (locs.length < 2) return locs[0];
    const rnd = random.randomInteger(0, locs.length - 1);
    return locs[rnd];
};

class SampleAverageAgent {
    private q: number[];
    private steps: number[];

    constructor(private k: number, private epsilon: number) {
        if (epsilon < 0 || epsilon > 1) throw new Error('Expected epsilon to be between 0 and 1');

        this.q = _.times(this.k, () => 0);
        this.steps = _.times(this.k, () => 0);
    }

    getAction(): number {
        const rnd = tf.randomUniform([1], 0, 1).get(0);
        if (rnd <= this.epsilon) return random.randomInteger(0, this.k - 1);

        const max = argMax(this.q);
        return max;
    }

    updateValueEstimates(a: number, r: number) {
        if (a > this.k || a < 0) throw new Error('Attempted to update action estimate outside of range');

        this.steps[a] += 1;
        this.q[a] = this.q[a] + (1 / this.steps[a]) * (r - this.q[a]);
    }

    reset() {
        this.q = _.times(this.k, () => 0);
        this.steps = _.times(this.k, () => 0);
    }
}

function evaluate(agent: SampleAverageAgent, env: KArmedBandit, steps: number) {
    agent.reset();
    const rewards = _.times(steps, () => {
        const a = agent.getAction();
        const reward = env.takeAction(a);
        agent.updateValueEstimates(a, reward);
        return reward;
    });

    return rewards;
}

async function run() {
    const steps = 1000;
    const runs = 200;
    const k = 10;

    const palette = createStandardPalette(3);

    const agents = [
        new SampleAverageAgent(k, 0),
        new SampleAverageAgent(k, 0.01),
        new SampleAverageAgent(k, 0.1),
    ];

    const plots = agents.map((agent) => {
        const rewards = _.times(runs, (run) => {
            console.log(run);
            const env = new KArmedBandit(k);

            return tf.tidy(() => evaluate(agent, env, steps));
        });

        const rewardMatrix = Matrix.fromData(rewards);
        const stats = describeColumns(rewardMatrix);

        const learningCurve = LineChart.fromArrayStats(stats);
        learningCurve.setColor(palette.next());

        return learningCurve;
    });

    await Remote.plot(combineTraces(plots, ''));
}

run()
    .catch(console.log);
