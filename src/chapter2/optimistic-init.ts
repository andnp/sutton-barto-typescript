import * as _ from 'lodash';
import { random } from 'mlts';
import { Matrix } from 'utilities-ts';
import { describeColumns, LineChart, Remote, createStandardPalette, combineTraces } from 'tsplot';

import { argMax } from '../utils/numerical';
import { randomNormal, randomUniform } from '../utils/statistical';


class KArmedBandit {
    private q_star: number[];
    constructor(private k: number) {
        this.q_star = _.times(this.k, () => randomNormal(0, 1));
    }

    takeAction(a: number): number {
        if (a < 0 || a >= this.k) throw new Error(`Action is outside of range. ${a}`);

        const value = this.q_star[a];
        const reward = randomNormal(value, 1);
        return reward;
    }
}

interface AgentSettings {
    optimisticInit: number;
    alpha: number;
    epsilon: number;
}

class OptimisticIncrementalAgent {
    private q: number[];

    constructor(readonly name: string, private k: number, private opts: AgentSettings) {
        if (this.opts.epsilon < 0 || this.opts.epsilon > 1) throw new Error('Expected epsilon to be between 0 and 1');

        this.q = _.times(this.k, () => this.opts.optimisticInit);
    }

    getAction(): number {
        const rnd = randomUniform(0, 1);
        if (rnd <= this.opts.epsilon) return random.randomInteger(0, this.k - 1);

        const max = argMax(this.q);
        return max;
    }

    updateValueEstimates(a: number, r: number) {
        if (a > this.k || a < 0) throw new Error('Attempted to update action estimate outside of range');

        this.q[a] = this.q[a] + this.opts.alpha * (r - this.q[a]);
    }

    reset() {
        this.q = _.times(this.k, () => this.opts.optimisticInit);
    }
}

function evaluate(agent: OptimisticIncrementalAgent, env: KArmedBandit, steps: number) {
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
        new OptimisticIncrementalAgent('optimistic', k, { optimisticInit: 5, alpha: 0.1, epsilon: 0 }),
        new OptimisticIncrementalAgent('epsilon greedy', k, { optimisticInit: 0, alpha: 0.1, epsilon: 0.1 }),
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

    await Remote.plot(combineTraces(plots));
}

run()
    .catch(console.log);
