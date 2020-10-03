import * as _ from 'lodash';
import { random } from 'mlts';
import { Matrix } from 'utilities-ts';
import { describeColumns, LineChart, Remote, createStandardPalette, combineTraces } from 'tsplot';

import { argMax } from '../utils/numerical';
import { randomNormal, randomUniform } from '../utils/statistical';

class KArmedBandit {
    private q_star: number[];
    constructor(private k: number) {
        this.q_star = _.times(this.k, () => 0);
    }

    takeAction(a: number): number {
        if (a < 0 || a >= this.k) throw new Error(`Action is outside of range. ${a}`);

        // take a random walk for each value
        this.q_star = this.q_star.map(value => randomNormal(value, 0.01));

        const value = this.q_star[a];
        const reward = randomNormal(value, 1);
        return reward;
    }
}

class SampleAverageAgent {
    private q: number[];

    constructor(private k: number, private alpha: number, private epsilon: number) {
        if (epsilon < 0 || epsilon > 1) throw new Error('Expected epsilon to be between 0 and 1');

        this.q = _.times(this.k, () => 0);
    }

    getAction(): number {
        const rnd = randomUniform(0, 1);
        if (rnd <= this.epsilon) return random.randomInteger(0, this.k - 1);

        const max = argMax(this.q);
        return max;
    }

    updateValueEstimates(a: number, r: number) {
        if (a > this.k || a < 0) throw new Error('Attempted to update action estimate outside of range');

        this.q[a] = this.q[a] + this.alpha * (r - this.q[a]);
    }

    reset() {
        this.q = _.times(this.k, () => 0);
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
    const steps = 10000;
    const runs = 200;
    const k = 10;

    const palette = createStandardPalette(3);

    const agents = [
        new SampleAverageAgent(k, 0.01, 0),
        new SampleAverageAgent(k, 0.01, 0.01),
        new SampleAverageAgent(k, 0.01, 0.1),
    ];

    const plots = agents.map((agent) => {
        const rewards = _.times(runs, (run) => {
            console.log(run);
            const env = new KArmedBandit(k);

            return evaluate(agent, env, steps);
        });

        const rewardMatrix = Matrix.fromData(rewards);
        const stats = describeColumns(rewardMatrix);

        const learningCurve = LineChart.fromArrayStats(stats);
        learningCurve.setColor(palette.next());

        return learningCurve;
    });

    await Remote.plot(combineTraces(plots));
}

run()
    .catch(console.log);
