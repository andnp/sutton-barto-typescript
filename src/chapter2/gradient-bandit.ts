import * as _ from 'lodash';
import { Matrix } from 'utilities-ts';
import { describeColumns, LineChart, Remote, createStandardPalette, combineTraces } from 'tsplot';

import { argMax } from '../utils/numerical';
import { randomNormal } from '../utils/statistical';


class KArmedBandit {
    private q_star: number[];
    constructor(private k: number) {
        this.q_star = _.times(this.k, () => randomNormal(4, 1));
    }

    takeAction(a: number): number {
        if (a < 0 || a >= this.k) throw new Error(`Action is outside of range. ${a}`);

        const value = this.q_star[a];
        const reward = randomNormal(value, 1);
        return reward;
    }
}

function softmax(arr: number[], a: number) {
    const sum = _.sum(arr.map(v => Math.exp(v)));
    return Math.exp(arr[a]) / sum;
}

interface AgentSettings {
    baseline: boolean;
    alpha: number;
}

class GradientBanditAgent {
    private h: number[];
    private r_bar: number;
    private t: number;

    constructor(readonly name: string, private k: number, private opts: AgentSettings) {
        this.h = _.times(this.k, () => 0);
        this.r_bar = 0;
        this.t = 0;
    }

    getAction(): number {
        const max = argMax(this.h);
        return max;
    }

    updateValueEstimates(a: number, r: number) {
        if (a > this.k || a < 0) throw new Error('Attempted to update action estimate outside of range');

        this.t++;
        this.r_bar = this.opts.baseline
            ? (this.r_bar * (this.t - 1) + r) / this.t
            : 0;

        this.h = this.h.map((pref, i) => {
            if (i === a) {
                return pref + this.opts.alpha * (r - this.r_bar) * (1 - softmax(this.h, a));
            } else {
                return pref - this.opts.alpha * (r - this.r_bar) * softmax(this.h, i);
            }
        });
    }

    reset() {
        this.h = _.times(this.k, () => 0);
        this.r_bar = 0;
        this.t = 0;
    }
}

function evaluate(agent: GradientBanditAgent, env: KArmedBandit, steps: number) {
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
    const runs = 1000;
    const k = 10;

    const palette = createStandardPalette(3);

    const agents = [
        new GradientBanditAgent('baseline', k, { baseline: true, alpha: 0.1}),
        new GradientBanditAgent('no baseline', k, { baseline: false, alpha: 0.1}),
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
        learningCurve.label(agent.name);

        return learningCurve;
    });

    await Remote.plot(combineTraces(plots, ''));
}

run()
    .catch(console.log);
