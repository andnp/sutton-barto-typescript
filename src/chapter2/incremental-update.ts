import * as tf from '@tensorflow/tfjs';
import * as _ from 'lodash';
import { random } from 'mlts';
import { Matrix, files } from 'utilities-ts';
import { describeColumns, LineChart, Remote, createStandardPalette, combineTraces } from 'tsplot';
import '@tensorflow/tfjs-node';
import { updateAtIndex } from '../utils/tfjs';

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

class SampleAverageAgent {
    private q: tf.Tensor1D;
    private steps: tf.Tensor1D;

    constructor(private k: number, private epsilon: number) {
        if (epsilon < 0 || epsilon > 1) throw new Error('Expected epsilon to be between 0 and 1');

        this.q = tf.zeros([this.k]);
        this.steps = tf.zeros([this.k]);
    }

    getAction(): number {
        return tf.tidy(() => {
            const rnd = tf.randomUniform([1], 0, 1).get(0);
            if (rnd <= this.epsilon) return random.randomInteger(0, this.k - 1);

            const max = this.q.argMax().get();
            return max;
        });
    }

    updateValueEstimates(a: number, r: number) {
        if (a > this.k || a < 0) throw new Error('Attempted to update action estimate outside of range');

        this.steps = updateAtIndex(this.steps, a, this.steps.get(a) + 1);
        this.q = updateAtIndex(this.q, a, this.q.get(a) + (1 / this.steps.get(a)) * (r - this.q.get(a)));
    }

    reset() {
        this.q = tf.zeros([this.k]);
        this.steps = tf.zeros([this.k]);
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
    const runs = 2000;
    const k = 10;

    const palette = createStandardPalette(3);

    const agents = [
        // new SampleAverageAgent(k, 0),
        // new SampleAverageAgent(k, 0.01),
        new SampleAverageAgent(k, 0.1),
        // new SampleAverageAgent(k, 0.1),
        // new SampleAverageAgent(k, 0.1),
    ];

    const plots = agents.map((agent) => {
        const rewards = _.times(runs, (run) => {
            console.log(run);
            const env = new KArmedBandit(k);

            return tf.tidy(() => evaluate(agent, env, steps));
        });

        const rewardMatrix = Matrix.fromData(rewards);
        const stats = describeColumns(rewardMatrix);
        const mean = stats.map(s => s.mean);

        // const learningCurve = LineChart.fromArrayStats(stats);
        const learningCurve = LineChart.fromArray(mean);
        learningCurve.setColor(palette.next());

        return learningCurve;
    });

    const combined = [] as LineChart[];
    for (let i = 0; i < plots.length; ++i) {
        const plot = plots[i];
        combined.push(plot);
        plot.smooth(false);
        plot.layout.yaxis = { range: [-1.1, 3.1] };
        const svg = await Remote.plot(combineTraces(combined));
        // await files.writeFile(`plot-${i}.svg`, svg);
        await files.writeFile('avg-2000.svg', svg);
    }

    // const svg = await Remote.plot(combineTraces(plots));
    // await files.writeFile('avg-20.svg', svg);
}

run()
    .catch(console.log);
